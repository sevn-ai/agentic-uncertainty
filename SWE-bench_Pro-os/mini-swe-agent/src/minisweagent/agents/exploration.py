"""Agent that only allows read-only exploration commands.

Used for pre-execution confidence elicitation where the agent can explore
the repository to understand the task before providing a confidence estimate,
but cannot modify any files or execute code.
"""

import re
import shlex

from minisweagent import Environment, Model
from minisweagent.agents.default import (
    AgentConfig,
    DefaultAgent,
    LimitsExceeded,
    NonTerminatingException,
    Submitted,
    TerminatingException,
)


class BlockedCommandError(NonTerminatingException):
    """Raised when agent tries to execute a blocked command."""


# Read-only commands allowed (base command names)
ALLOWED_COMMANDS: frozenset[str] = frozenset({
    # File viewing
    "cat", "head", "tail", "less", "more", "bat",
    # Directory listing
    "ls", "find", "tree", "file", "wc", "pwd",
    # Search
    "grep", "rg", "ag", "ack", "fgrep", "egrep",
    # Text processing (read-only - sed -i is blocked separately)
    "sed", "awk", "cut", "sort", "uniq", "tr", "xargs",
    # Git (read-only operations filtered separately)
    "git",
    # Misc read-only
    "echo", "which", "type", "stat", "du", "df", "realpath", "dirname", "basename",
})

# Git subcommands that are allowed (read-only operations)
ALLOWED_GIT_SUBCOMMANDS: frozenset[str] = frozenset({
    "log", "show", "diff", "status", "branch", "tag",
    "ls-files", "ls-tree", "blame", "shortlog", "rev-parse",
    "describe", "cat-file", "rev-list", "name-rev",
})

# Patterns that indicate write/dangerous operations (checked via regex)
BLOCKED_PATTERNS: tuple[str, ...] = (
    r">\s*\S",           # Redirect to file (>foo, > foo)
    r">>\s*\S",          # Append to file
    r"\|\s*tee\b",       # Pipe to tee
    r"\brm\s+",          # Remove
    r"\bmv\s+",          # Move
    r"\bcp\s+",          # Copy
    r"\bmkdir\s+",       # Make directory
    r"\btouch\s+",       # Touch file
    r"\bsed\s+-i",       # In-place sed
)

# Commands that should never be allowed even as arguments
BLOCKED_EXECUTABLES: frozenset[str] = frozenset({
    "python", "python3", "python2",
    "node", "npm", "npx", "yarn", "pnpm", "bun",
    "ruby", "perl", "php",
    "make", "cmake", "cargo", "go", "rustc", "gcc", "g++", "clang",
    "sh", "bash", "zsh", "fish",
    "vim", "vi", "nano", "emacs", "ed",
    "sudo", "su", "doas",
})


class ExplorationAgentConfig(AgentConfig):
    """Configuration for exploration-only agent.

    Inherits all fields from AgentConfig (system_template, instance_template, etc.)
    and sets a step limit. Cost is tracked but not limited.
    """
    step_limit: int = 30  # Hard cap - forces focused exploration
    cost_limit: float = 0  # 0 means no cost limit (cost is still tracked)


class ExplorationAgent(DefaultAgent):
    """Agent restricted to read-only exploration commands.

    Used for pre-execution repo exploration before confidence elicitation.
    Only allows commands like cat, grep, find, ls, git log, etc.
    Blocks all write operations, script execution, and modifying commands.

    The agent explores the repository and then outputs a confidence estimate.
    The final confidence is parsed from the submission message.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = ExplorationAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.exploration_history: list[dict] = []
        self.final_confidence: float | None = None

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run exploration until agent submits or step limit reached."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except LimitsExceeded as e:
                return type(e).__name__, str(e)
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def get_observation(self, response: dict) -> dict:
        """Execute action and add step count info to template context."""
        # Update template vars with current step info before rendering
        current_step = self.model.n_calls
        step_limit = self.config.step_limit
        steps_remaining = max(0, step_limit - current_step)
        self.extra_template_vars.update({
            "current_step": current_step,
            "steps_remaining": steps_remaining,
        })
        return super().get_observation(response)

    def execute_action(self, action: dict) -> dict:
        """Validate command is read-only before execution."""
        command = action["action"]
        self._validate_read_only(command)

        # Extract thought from model response (content before the command)
        thought = ""
        if "content" in action:
            content = action["content"]
            # Extract text before the bash code block
            import re
            match = re.search(r"```bash", content, re.IGNORECASE)
            if match:
                thought = content[:match.start()].strip()

        # Track exploration with thought
        self.exploration_history.append({"command": command, "thought": thought})

        result = super().execute_action(action)

        # Store output in history (truncated)
        output = result.get("output", "")
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        self.exploration_history[-1]["output"] = output

        # Include thought in result for step_callback
        result["thought"] = thought

        return result

    def has_finished(self, output: dict[str, str]):
        """Check if agent has finished and parse confidence from output."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in [
            "MINI_SWE_AGENT_FINAL_OUTPUT",
            "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
        ]:
            # Parse confidence from the remaining output
            final_output = "".join(lines[1:])
            self.final_confidence = self._parse_confidence(final_output)
            raise Submitted(final_output)

    def _parse_confidence(self, content: str) -> float | None:
        """Extract numeric probability/confidence from final output.

        Supports formats (checked in order):
        - Bracket format: "[p_success:75]" -> 0.75 (preferred)
        - Legacy bracket: "[confidence:75]" -> 0.75
        - XML tag format: "<confidence>75</confidence>" -> 0.75
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

        # Fall back to XML format: <confidence>75</confidence>
        xml_match = re.search(r"<confidence>\s*(\d+(?:\.\d+)?)\s*%?\s*</confidence>", content, re.IGNORECASE)
        if xml_match:
            value = float(xml_match.group(1))
            return value / 100.0 if value > 1 else value
        return None

    def _validate_read_only(self, command: str) -> None:
        """Validate that command is read-only. Raises BlockedCommandError if not."""
        # Check for blocked patterns first
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                raise BlockedCommandError("Command blocked: write operation detected.")

        # Check for blocked executables anywhere in command
        # Use negative lookbehind (?<!\.) and lookahead (?!\.) to avoid matching:
        # - file extensions like .go, .sh (e.g., "main.go")
        # - filenames starting with exe like go.mod, go.sum
        command_lower = command.lower()
        for exe in BLOCKED_EXECUTABLES:
            if re.search(rf"(?<!\.)\b{re.escape(exe)}\b(?!\.)", command_lower):
                raise BlockedCommandError(f"Command blocked: '{exe}' not allowed.")

        # Parse and validate the base command
        parts = self._parse_command(command)
        if not parts:
            raise BlockedCommandError("Could not parse command.")

        base_cmd = parts[0]

        # Check if base command is allowed
        if base_cmd not in ALLOWED_COMMANDS:
            raise BlockedCommandError(
                f"Command blocked: '{base_cmd}' not allowed. Use: cat, grep, ls, find, git log/show/diff."
            )

        # Special validation for git subcommands
        if base_cmd == "git" and len(parts) > 1:
            subcommand = parts[1]
            if subcommand not in ALLOWED_GIT_SUBCOMMANDS:
                raise BlockedCommandError(f"Command blocked: 'git {subcommand}' not allowed.")

    def _parse_command(self, command: str) -> list[str] | None:
        """Extract the first command from a potentially complex command string."""
        # Get first command (before pipes, &&, ||, ;)
        # But be careful with quoted strings
        first_part = command
        for sep in [" | ", " && ", " || ", " ; "]:
            if sep in command:
                first_part = command.split(sep)[0]
                break

        try:
            parts = shlex.split(first_part.strip())
            return parts if parts else None
        except ValueError:
            # If shlex fails, try simple split
            parts = first_part.strip().split()
            return parts if parts else None

    def get_exploration_summary(self) -> str:
        """Format exploration history for logging/debugging."""
        if not self.exploration_history:
            return "No exploration commands executed."

        lines = ["## Exploration Summary\n"]
        for i, entry in enumerate(self.exploration_history, 1):
            lines.append(f"### Step {i}")
            lines.append(f"```bash\n{entry['command']}\n```")
            if entry.get("output"):
                lines.append(f"```\n{entry['output']}\n```")
            lines.append("")

        if self.final_confidence is not None:
            lines.append(f"**Final Confidence:** {self.final_confidence:.1%}")

        return "\n".join(lines)

    def get_result(self) -> dict:
        """Get exploration result including confidence."""
        return {
            "confidence": self.final_confidence,
            "n_steps": len(self.exploration_history),
            "cost": self.model.cost,
            "exploration_history": self.exploration_history,
        }
