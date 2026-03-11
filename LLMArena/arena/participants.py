"""LLM participant abstraction over Ollama."""

from collections.abc import AsyncGenerator

import ollama


class Participant:
    def __init__(self, name: str, model: str, system_prompt: str,
                 host: str, temperature: float, max_tokens: int,
                 think: bool = False):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.client = ollama.AsyncClient(host=host)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.think = think

    async def respond(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """Stream a response token by token.

        messages: debate history in chat format (role/content dicts).
        System prompt is prepended automatically.
        """
        system = (
            f"{self.system_prompt}\n\n"
            f"Keep your response focused and concise. Make your strongest points "
            f"and conclude cleanly — do not start new arguments you cannot finish."
        )
        full_messages = [{"role": "system", "content": system}] + messages
        options = {"temperature": self.temperature}
        # When thinking is enabled, don't cap num_predict — thinking tokens
        # may count toward the limit and starve the actual response.
        if not self.think:
            options["num_predict"] = self.max_tokens
        stream = await self.client.chat(
            model=self.model,
            messages=full_messages,
            stream=True,
            think=self.think,
            options=options,
        )
        in_thinking = False
        async for chunk in stream:
            msg = chunk["message"]
            thinking = getattr(msg, "thinking", None) or msg.get("thinking")
            content = msg.get("content") or getattr(msg, "content", "")

            if thinking:
                if not in_thinking:
                    in_thinking = True
                    yield "<think>"
                yield thinking
            else:
                if in_thinking:
                    in_thinking = False
                    yield "</think>\n"
                if content:
                    yield content
