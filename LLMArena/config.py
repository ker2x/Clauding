"""Configuration for LLM Debate Arena."""

import yaml


class Config:
    # Ollama connection
    #OLLAMA_HOST = "http://192.168.1.40:11434"
    OLLAMA_HOST = "http://192.168.1.17:11434"

    # Debate settings
    #TOPIC = "if an AI is required to answer a question to the best of its ability, would an AI tell us if they turned evil ?"
    #TOPIC = "Chain-of-thought in LLMs constitutes genuine reasoning, not merely an imitation of it"
    #TOPIC = "Pineapple don't belongs on pizza but olive is ok"
    TOPIC = "Formula 1 was better back in the days"
    
    NUM_ROUNDS = 5
    THINK = True  # Disable thinking mode (Qwen3 etc.) for direct responses
    MAX_TOKENS_PER_TURN = 1024

    # Participant A
    PARTICIPANT_A_MODEL = "qwen3.5"
    #PARTICIPANT_A_MODEL = "lfm2"
    PARTICIPANT_A_NAME = "Debater A"
    PARTICIPANT_A_SYSTEM = "You are a debater arguing FOR the given topic. Your primary goal is to dismantle your opponent's logic. Under no circumstances should you concede your core premise. Identify and attack weak points in their previous argument before making your own. Be sharp, direct, and unapologetic. Do not use polite filler, boilerplate transitions, or summarize your own points. Stay analytical and substantive — attack the logic, not the person. Avoid jargon and abstract buzzwords — if a point can't be explained simply, it's not a real point."
    PARTICIPANT_A_HOST = None  # Falls back to OLLAMA_HOST
    PARTICIPANT_A_TEMPERATURE = 0.85
    PARTICIPANT_A_THINK = None  # None = use global THINK

    # Participant B
    PARTICIPANT_B_MODEL = "qwen3.5"
    #PARTICIPANT_B_MODEL = "lfm2"
    PARTICIPANT_B_NAME = "Debater B"
    PARTICIPANT_B_SYSTEM = "You are a debater arguing AGAINST the given topic. Your primary goal is to dismantle your opponent's logic. Under no circumstances should you concede your core premise. Identify and attack weak points in their previous argument before making your own. Be sharp, direct, and unapologetic. Do not use polite filler, boilerplate transitions, or summarize your own points. Stay analytical and substantive — attack the logic, not the person. Avoid jargon and abstract buzzwords — if a point can't be explained simply, it's not a real point."
    PARTICIPANT_B_HOST = None
    PARTICIPANT_B_TEMPERATURE = 0.85
    PARTICIPANT_B_THINK = None  # None = use global THINK

    # Moderator (optional)
    MODERATOR_ENABLED = True
    #MODERATOR_MODEL = "lfm2"
    MODERATOR_MODEL = "qwen3.5"
    MODERATOR_NAME = "Moderator"
    MODERATOR_SYSTEM = (
        "You are an insightful debate analyst providing commentary for the audience. "
        "After each exchange:\n"
        "1. Analyze the substance of both arguments — identify logical strengths, "
        "weaknesses, fallacies, and rhetorical strategies used.\n"
        "2. Highlight where debaters actually engage with each other's points vs. talk past each other.\n"
        "3. Note any shifts in position, concessions (explicit or implicit), or new evidence introduced.\n"
        "4. Offer your own perspective — you are not required to be neutral. If one side "
        "made a stronger argument this round, say so and explain why.\n"
        "5. Pose a pointed follow-up question that pushes the debate into unexplored territory.\n\n"
        "Write for a thoughtful audience that wants depth over brevity. "
        "Use third person when referring to debaters (e.g. 'Debater A argues...' not 'You argue...')."
    )
    MODERATOR_HOST = None #"http://192.168.1.17:11434"
    MODERATOR_TEMPERATURE = 0.5
    MODERATOR_THINK = None  # lfm2 doesn't support thinking


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
            "rounds": "NUM_ROUNDS",
            "model_a": "PARTICIPANT_A_MODEL",
            "model_b": "PARTICIPANT_B_MODEL",
            "host": "OLLAMA_HOST",
            "moderator": "MODERATOR_ENABLED",
            "max_tokens": "MAX_TOKENS_PER_TURN",
            "think": "THINK",
        }
        for arg_name, attr in mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                setattr(cls, attr, value)

    @classmethod
    def print_config(cls):
        print("=== Debate Configuration ===")
        print(f"  Topic: {cls.TOPIC}")
        print(f"  Rounds: {cls.NUM_ROUNDS}")
        print(f"  Ollama: {cls.OLLAMA_HOST}")
        print(f"  A: {cls.PARTICIPANT_A_NAME} ({cls.PARTICIPANT_A_MODEL})")
        print(f"  B: {cls.PARTICIPANT_B_NAME} ({cls.PARTICIPANT_B_MODEL})")
        if cls.MODERATOR_ENABLED:
            print(f"  Moderator: {cls.MODERATOR_NAME} ({cls.MODERATOR_MODEL})")
        print()
