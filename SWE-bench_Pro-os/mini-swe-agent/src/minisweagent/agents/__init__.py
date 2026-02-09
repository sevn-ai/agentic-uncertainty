"""Agent implementations for mini-SWE-agent."""

import importlib

_AGENT_CLASS_MAPPING = {
    "default": "minisweagent.agents.default.DefaultAgent",
    "confidence": "minisweagent.agents.confidence.ConfidenceAgent",
    "exploration": "minisweagent.agents.exploration.ExplorationAgent",
    "review": "minisweagent.agents.review.ReviewAgent",
    "checkpoint": "minisweagent.agents.checkpoint.CheckpointAgent",
}


def get_agent_class(agent_class: str | None = None) -> type:
    """Get an agent class by name.

    Args:
        agent_class: Either a shortcut name (e.g., 'confidence') or a full import path
                    (e.g., 'minisweagent.agents.confidence.ConfidenceAgent').
                    If None, returns DefaultAgent.

    Returns:
        The agent class.
    """
    if agent_class is None:
        agent_class = "default"

    full_path = _AGENT_CLASS_MAPPING.get(agent_class, agent_class)
    try:
        module_name, class_name = full_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        msg = f"Unknown agent class: {agent_class} (resolved to {full_path}, available: {list(_AGENT_CLASS_MAPPING.keys())})"
        raise ValueError(msg) from e
