"""Stage 4: Fine-tune Qwen2.5-0.5B on distilled dataset."""

import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config

CHAT_TEMPLATE = (
    "<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n{answer}<|im_end|>"
)


def load_dataset(config: Config) -> list[dict]:
    """Load training examples from dataset.jsonl."""
    examples = []
    with open(config.DATASET_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict) -> str:
    """Format a single example as a chat string."""
    return CHAT_TEMPLATE.format(
        prompt=example["prompt"],
        reasoning=example["reasoning"],
        answer=example["answer"],
    )


def compute_prompt_len(tokenizer, example: dict) -> int:
    """Compute the number of tokens in the prompt portion (to mask from loss)."""
    prompt_str = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    return len(tokenizer.encode(prompt_str, add_special_tokens=False))


def prepare_batch(
    examples: list[dict],
    tokenizer,
    max_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize and pad a batch, returning input_ids and labels (with prompt masked)."""
    all_input_ids = []
    all_labels = []

    for example in examples:
        text = format_example(example)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        prompt_len = compute_prompt_len(tokenizer, example)

        # Truncate
        tokens = tokens[:max_seq_len]

        # Labels: -100 for prompt tokens, actual token ids for response
        labels = [-100] * min(prompt_len, len(tokens)) + tokens[prompt_len:]

        all_input_ids.append(tokens)
        all_labels.append(labels)

    # Pad to fixed max_seq_len (not batch max) — MPS caches Metal buffers per
    # tensor shape, so variable lengths cause unbounded driver memory growth.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_ids = []
    padded_labels = []
    attention_masks = []
    for ids, lab in zip(all_input_ids, all_labels):
        pad_len = max_seq_len - len(ids)
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_labels.append(lab + [-100] * pad_len)
        attention_masks.append([1] * len(ids) + [0] * pad_len)

    return (
        torch.tensor(padded_ids, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
        torch.tensor(attention_masks, dtype=torch.long, device=device),
    )


def evaluate(
    model,
    tokenizer,
    eval_examples: list[dict],
    config: Config,
    device: torch.device,
) -> dict:
    """Evaluate exact-match accuracy on eval set."""
    model.eval()
    correct = 0
    total = len(eval_examples)

    with torch.no_grad():
        for example in eval_examples:
            prompt_str = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
            attention_mask = torch.ones_like(input_ids)
            eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)

            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=eos_token_id,
            )

            generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Extract answer: after </think> tag, or use answer extraction heuristics
            if "</think>" in generated:
                answer_part = generated.split("</think>")[-1].strip()
            else:
                answer_part = generated.strip()

            # Try to extract a number from the answer
            import re
            match = re.search(r"(-?\d+)", answer_part)
            if match and match.group(1) == example["answer"]:
                correct += 1
            elif answer_part == example["answer"]:
                correct += 1

    model.train()
    return {"accuracy": correct / total if total > 0 else 0, "correct": correct, "total": total}


def train(config: Config, resume_from: str | None = None) -> None:
    """Main training loop."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_path = resume_from or config.STUDENT_MODEL
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to(device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and split dataset
    all_examples = load_dataset(config)
    random.seed(config.SEED)
    random.shuffle(all_examples)

    eval_size = max(1, int(len(all_examples) * config.EVAL_SPLIT))
    eval_examples = all_examples[:eval_size]
    train_examples = all_examples[eval_size:]
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")

    # Filter examples that are too long
    filtered_train = []
    for ex in train_examples:
        text = format_example(ex)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= config.MAX_SEQ_LEN:
            filtered_train.append(ex)
    if len(filtered_train) < len(train_examples):
        print(f"Filtered {len(train_examples) - len(filtered_train)} examples exceeding max_seq_len")
        train_examples = filtered_train

    # Optimizer — use SGD to save memory (AdamW stores 2 extra copies of params)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=0.9,
        weight_decay=config.WEIGHT_DECAY,
    )

    # LR schedule
    steps_per_epoch = math.ceil(len(train_examples) / config.BATCH_SIZE)
    total_steps = steps_per_epoch * config.NUM_EPOCHS // config.GRAD_ACCUM_STEPS
    warmup_steps = config.WARMUP_STEPS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Training
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(config.NUM_EPOCHS):
        random.shuffle(train_examples)
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            range(0, len(train_examples), config.BATCH_SIZE),
            desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}",
        )
        for batch_start in pbar:
            batch = train_examples[batch_start : batch_start + config.BATCH_SIZE]
            input_ids, labels, attention_mask = prepare_batch(batch, tokenizer, config.MAX_SEQ_LEN, device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_val = outputs.loss.item()
            loss = outputs.loss / config.GRAD_ACCUM_STEPS
            loss.backward()
            del outputs, loss, input_ids, labels, attention_mask

            epoch_loss += loss_val
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}", step=global_step, lr=f"{scheduler.get_last_lr()[0]:.1e}")

            if num_batches % config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.CHECKPOINT_EVERY == 0:
                    ckpt_path = config.CHECKPOINT_DIR / f"step_{global_step}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    tqdm.write(f"Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_loss = epoch_loss / max(1, num_batches)
        print(f"\nEpoch {epoch + 1} done | avg loss: {avg_loss:.4f}")

        # Eval
        eval_results = evaluate(model, tokenizer, eval_examples, config, device)
        print(f"Eval: {eval_results['correct']}/{eval_results['total']} = {eval_results['accuracy']:.2%}\n")

    # Save final
    final_path = config.CHECKPOINT_DIR / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Training complete. Final model saved to {final_path}")
