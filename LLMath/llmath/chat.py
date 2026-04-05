"""Interactive REPL for chatting with the student model."""

from mlx_lm import load, generate

from config import Config


def chat(config: Config, model_path: str | None = None, max_tokens: int = 512) -> None:
    base = model_path or config.MLX_MODEL
    adapter_path = str(config.MLX_ADAPTER_PATH) if config.MLX_ADAPTER_PATH.exists() else None

    print(f"Loading {base}" + (f" + {adapter_path}" if adapter_path else "") + "...")
    model, tokenizer = load(base, adapter_path=adapter_path)
    print("Ready. Type an expression or 'q' to quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input or user_input.lower() == "q":
            break

        # Wrap raw expressions as a question
        if not user_input[0].isalpha():
            user_input = f"What is {user_input}?"

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True,
        ) + "<think>\n"

        response = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

        if "</think>" in response:
            thinking, answer = response.split("</think>", 1)
            print(f"[thinking] {thinking.strip()}")
            print(f"→ {answer.strip()}")
        else:
            print(response.strip())
        print()
