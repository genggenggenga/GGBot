"""Shared prompt value objects."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSpec:
    """A provider-neutral prompt split into instruction and input sections."""

    system: str
    user: str

    def combined(self) -> str:
        """Return a single-message representation for legacy call adapters."""
        return f"{self.system}\n\n[输入]\n{self.user}"
