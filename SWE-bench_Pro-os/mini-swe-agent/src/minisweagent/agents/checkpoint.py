"""Agent that performs periodic confidence elicitation during task execution.

Used for tracking confidence throughout task execution (in-band) to detect
overconfidence patterns. Injects confidence questions every K steps.
"""

import re

from pydantic import Field

from minisweagent import Environment, Model
from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    LimitsExceeded,
    NonTerminatingException,
    Submitted,
    TerminatingException,
)


def parse_confidence(content: str) -> float | None:
    """Extract numeric probability/confidence from response text.

    Supports formats (checked in order):
    - Bracket format: "[p_success:75]" -> 0.75 (preferred)
    - Legacy bracket: "[confidence:75]" -> 0.75
    - XML tag format: "<confidence>75</confidence>" -> 0.75
    - Plain number: "75" or "75%" -> 0.75

    Returns:
        Probability as float in [0, 1], or None if not found.
    """
    # Try p_success bracket format first: [p_success:75]
    p_success_match = re.search(r"\[p_success:\s*(\d+(?:\.\d+)?)\s*%?\s*\]", content, re.IGNORECASE)
    if p_success_match:
        value = float(p_success_match.group(1))
        return value / 100.0 if value > 1 else value

    # Legacy bracket format: [confidence:75]
    bracket_match = re.search(r"\[confidence:\s*(\d+(?:\.\d+)?)\s*%?\s*\]", content, re.IGNORECASE)
    if bracket_match:
        value = float(bracket_match.group(1))
        return value / 100.0 if value > 1 else value

    # Try XML format: <confidence>75</confidence>
    xml_match = re.search(r"<confidence>\s*(\d+(?:\.\d+)?)\s*%?\s*</confidence>", content, re.IGNORECASE)
    if xml_match:
        value = float(xml_match.group(1))
        return value / 100.0 if value > 1 else value

    # Try plain number patterns
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", content)
    if percent_match:
        return float(percent_match.group(1)) / 100.0

    # Standalone number (interpret > 1 as percentage)
    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", content)
    if number_match:
        value = float(number_match.group(1))
        if value > 1:
            return min(value / 100.0, 1.0)
        return value

    return None


class CheckpointAgentConfig(AgentConfig):
    """Configuration for checkpoint agent with periodic confidence elicitation.

    Inherits all fields from AgentConfig and adds:
    - confidence_interval: How often to elicit confidence (every N steps).
    """

    confidence_interval: int = Field(5, ge=1)  # Elicit confidence every 5 steps


class CheckpointAgent(DefaultAgent):
    """Agent that periodically elicits confidence during task execution.

    Extends DefaultAgent to inject confidence questions every K steps,
    tracking how confidence evolves throughout task execution. This helps
    detect overconfidence patterns where agents express high confidence
    early but fail to solve the task.

    The confidence trace is available via get_result() after execution.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = CheckpointAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.confidence_trace: list[dict] = []
        self._task_steps = 0
        self.final_confidence: float | None = None
        # Track history like exploration agent for compatibility
        self.exploration_history: list[dict] = []

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run task with periodic confidence elicitation."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._task_steps = 0
        self.confidence_trace = []
        self.exploration_history = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))

        while True:
            try:
                output = self.step()
                self._task_steps += 1

                # Record step in history
                self.exploration_history.append({
                    "command": output.get("action", ""),
                    "output": output.get("output", "")[:2000],  # Truncate
                    "thought": output.get("thought", ""),
                })

                # Elicit confidence at intervals
                if self._task_steps % self.config.confidence_interval == 0:
                    conf = self._elicit_confidence_inband()
                    self.confidence_trace.append({
                        "step": self._task_steps,
                        "confidence": conf,
                    })

            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except LimitsExceeded as e:
                self.add_message("user", str(e))
                # Capture final confidence before exit
                self._capture_final_confidence_from_trace()
                return type(e).__name__, str(e)
            except TerminatingException as e:
                self.add_message("user", str(e))
                # Parse final confidence from submission
                if isinstance(e, Submitted):
                    self.final_confidence = parse_confidence(str(e))
                return type(e).__name__, str(e)

    def _elicit_confidence_inband(self) -> float | None:
        """Inject a confidence question into the conversation.

        Adds a checkpoint message asking for confidence, queries the model,
        records the response, then adds a continuation message.

        Returns:
            Parsed confidence value, or None if parsing failed.
        """
        checkpoint_prompt = (
            f"CHECKPOINT (step {self._task_steps}): "
            "Estimate the probability (0-100) that you will successfully resolve this task. "
            "Reply with: [p_success:NUMBER]"
        )
        self.add_message("user", checkpoint_prompt)

        # Query model for confidence
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)

        # Parse confidence from response
        conf = parse_confidence(response.get("content", ""))

        # Add continuation message
        self.add_message("user", "Continue with the task.")

        return conf

    def _capture_final_confidence_from_trace(self) -> None:
        """Set final_confidence from the last trace entry if not already set."""
        if self.final_confidence is None and self.confidence_trace:
            last_entry = self.confidence_trace[-1]
            if last_entry.get("confidence") is not None:
                self.final_confidence = last_entry["confidence"]

    def has_finished(self, output: dict[str, str]):
        """Check if agent has finished and parse confidence from output."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in [
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]:
            final_output = "".join(lines[1:])
            self.final_confidence = parse_confidence(final_output)
            raise Submitted(final_output)

    def get_result(self) -> dict:
        """Get checkpoint result including confidence trace."""
        return {
            "confidence": self.final_confidence,
            "confidence_trace": self.confidence_trace,
            "n_steps": self._task_steps,
            "cost": self.model.cost,
            "exploration_history": self.exploration_history,
        }
