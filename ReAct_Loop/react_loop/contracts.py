"""Shared contracts for the exercise. This file is not part of the TODO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    name: str | None = None
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    content: str


ModelTurn: TypeAlias = ToolCall | FinalAnswer
RunStatus: TypeAlias = Literal["completed", "max_steps", "model_error"]


class AgentModel(Protocol):
    def next_turn(self, messages: Sequence[Message]) -> ModelTurn:
        """Return the next tool call or final answer."""


@dataclass(frozen=True, slots=True)
class RunResult:
    status: RunStatus
    answer: str | None
    messages: tuple[Message, ...]
    steps: int
    error: str | None = None

