"""Configuration for Chain-of-Debate Arena."""

import yaml


class Config:
    # Ollama connection
    OLLAMA_HOST = "http://192.168.1.40:11434"
    #OLLAMA_HOST = "http://192.168.1.17:11434"

    # Session settings
    TOPIC = ""  # Empty = wait for user input
    THINK = True
    MAX_TOKENS_PER_TURN = 1024

    # Thinker A
    PARTICIPANT_A_MODEL = "gemma3:12b"
    PARTICIPANT_A_NAME = "Thinker A"
    PARTICIPANT_A_SYSTEM = (
        "You are Thinker A. A moderator will assign you a specific role for each question. "
        "When you see the moderator's message, find the role assigned to Thinker A "
        "and adopt it fully. Ignore any role assigned to Thinker B — that is for a "
        "different AI. Answer the user's question from your assigned role only. "
        "Be concrete, use examples. Avoid jargon and filler."
    )
    PARTICIPANT_A_HOST = None  # Falls back to OLLAMA_HOST
    PARTICIPANT_A_TEMPERATURE = 0.85
    PARTICIPANT_A_THINK = False  # None = use global THINK

    # Thinker B
    PARTICIPANT_B_MODEL = "lfm2"
    PARTICIPANT_B_NAME = "Thinker B"
    PARTICIPANT_B_SYSTEM = (
        "You are Thinker B. A moderator will assign you a specific role for each question. "
        "When you see the moderator's message, find the role assigned to Thinker B "
        "and adopt it fully. Ignore any role assigned to Thinker A — that is for a "
        "different AI. Answer the user's question from your assigned role only. "
        "Be concrete, use examples. Avoid jargon and filler."
    )
    PARTICIPANT_B_HOST = "http://192.168.1.17:11434"
    PARTICIPANT_B_TEMPERATURE = 0.85
    PARTICIPANT_B_THINK = False  # None = use global THINK

    # Moderator (orchestrator)
    MODERATOR_ENABLED = True
    MODERATOR_MODEL = "qwen3.5"
    MODERATOR_NAME = "Moderator"
    MODERATOR_SYSTEM = (
        "You are a coordinator. You work with two separate AI thinkers named "
        "Thinker A and Thinker B. They are independent AI models that will each "
        "receive your instructions and respond separately — you do not simulate them. "
        "You will be called at different stages and told what to do each time. "
        "Follow the specific instruction you are given for each turn. "
        "Each instruction is self-contained — only follow the current instruction, "
        "not constraints from previous turns."
    )
    MODERATOR_HOST = None
    MODERATOR_TEMPERATURE = 0.5
    MODERATOR_THINK = None

    # Nested YAML key mapping
    _YAML_NESTED = {
        "participant_a": "PARTICIPANT_A",
        "participant_b": "PARTICIPANT_B",
        "moderator": "MODERATOR",
    }

    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)
        if not data:
            return
        for key, value in data.items():
            if key in cls._YAML_NESTED and isinstance(value, dict):
                prefix = cls._YAML_NESTED[key]
                for subkey, subvalue in value.items():
                    attr = f"{prefix}_{subkey.upper()}"
                    if hasattr(cls, attr):
                        setattr(cls, attr, subvalue)
            else:
                attr = key.upper()
                if hasattr(cls, attr):
                    setattr(cls, attr, value)

    @classmethod
    def apply_cli_overrides(cls, args):
        mapping = {
            "topic": "TOPIC",
            "model_a": "PARTICIPANT_A_MODEL",
            "model_b": "PARTICIPANT_B_MODEL",
            "host": "OLLAMA_HOST",
            "max_tokens": "MAX_TOKENS_PER_TURN",
            "think": "THINK",
        }
        for arg_name, attr in mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                setattr(cls, attr, value)

    @classmethod
    def print_config(cls):
        print("=== Chain-of-Debate Configuration ===")
        if cls.TOPIC:
            print(f"  Topic: {cls.TOPIC}")
        else:
            print("  Topic: (waiting for user input)")
        print(f"  Ollama: {cls.OLLAMA_HOST}")
        print(f"  A: {cls.PARTICIPANT_A_NAME} ({cls.PARTICIPANT_A_MODEL})")
        print(f"  B: {cls.PARTICIPANT_B_NAME} ({cls.PARTICIPANT_B_MODEL})")
        print(f"  Moderator: {cls.MODERATOR_NAME} ({cls.MODERATOR_MODEL})")
        print()
