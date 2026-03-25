#!/usr/bin/env python3
"""REPL: Chat with the student model."""

import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Model path (default: base from config)")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    config = Config()
    model_path = args.model or config.STUDENT_MODEL
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16).to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Ready. Type an expression or 'q' to quit.\n")

    while True:
        try:
            expr = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not expr or expr.lower() == "q":
            break

        # If user types a raw expression, wrap it; otherwise use as-is
        if not expr[0].isalpha():
            expr = f"What is {expr}?"
        prompt = f"<|im_start|>user\n{expr}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)

        # Use <|im_end|> as stop token
        eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        # Show reasoning and answer
        if "</think>" in response:
            thinking, answer = response.split("</think>", 1)
            print(f"[thinking] {thinking.strip()}")
            print(f"→ {answer.strip()}")
        else:
            print(response.strip())
        print()


if __name__ == "__main__":
    main()
