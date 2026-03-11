"""Factory functions for creating debate participants."""

from arena.participants import Participant


def _resolve_think(per_participant, config) -> bool:
    """Per-participant THINK overrides global THINK if set."""
    return per_participant if per_participant is not None else config.THINK


def make_participant(prefix: str, config) -> Participant:
    """Create a participant from config. prefix is 'A' or 'B'."""
    think = _resolve_think(
        getattr(config, f"PARTICIPANT_{prefix}_THINK", None), config
    )
    return Participant(
        name=getattr(config, f"PARTICIPANT_{prefix}_NAME"),
        model=getattr(config, f"PARTICIPANT_{prefix}_MODEL"),
        system_prompt=getattr(config, f"PARTICIPANT_{prefix}_SYSTEM"),
        host=getattr(config, f"PARTICIPANT_{prefix}_HOST") or config.OLLAMA_HOST,
        temperature=getattr(config, f"PARTICIPANT_{prefix}_TEMPERATURE"),
        max_tokens=config.MAX_TOKENS_PER_TURN,
        think=think,
    )


def make_moderator(config) -> Participant | None:
    """Create a moderator if enabled, else None."""
    if not config.MODERATOR_ENABLED:
        return None
    think = _resolve_think(
        getattr(config, "MODERATOR_THINK", None), config
    )
    return Participant(
        name=config.MODERATOR_NAME,
        model=config.MODERATOR_MODEL,
        system_prompt=config.MODERATOR_SYSTEM,
        host=config.MODERATOR_HOST or config.OLLAMA_HOST,
        temperature=config.MODERATOR_TEMPERATURE,
        max_tokens=config.MAX_TOKENS_PER_TURN,
        think=think,
    )
