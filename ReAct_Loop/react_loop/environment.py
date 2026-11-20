"""Deterministic model and tool environment used by the exercise tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from .contracts import Message, ModelTurn, ToolCall


class UnknownToolError(LookupError):
    """Raised when the model requests a tool that is not registered."""


class ToolExecutionError(RuntimeError):
    """Raised when a registered tool fails while executing."""


Tool = Callable[..., object]


class ToolRegistry:
    """A minimal named tool dispatcher."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.calls: list[ToolCall] = []

    def register(self, name: str, tool: Tool) -> None:
        if not name:
            raise ValueError("tool name cannot be empty")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = tool

    def invoke(self, call: ToolCall) -> object:
        self.calls.append(call)
        tool = self._tools.get(call.name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {call.name}")
        try:
            return tool(**dict(call.arguments))
        except Exception as exc:
            raise ToolExecutionError(
                f"tool {call.name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc


class ScriptedModel:
    """Returns predefined turns, making loop behavior deterministic."""

    def __init__(self, turns: Iterable[ModelTurn | Exception | object]) -> None:
        self._turns = iter(turns)
        self.histories: list[tuple[Message, ...]] = []

    def next_turn(self, messages: Sequence[Message]) -> ModelTurn:
        self.histories.append(tuple(messages))
        try:
            turn = next(self._turns)
        except StopIteration as exc:
            raise RuntimeError("scripted model has no turn left") from exc
        if isinstance(turn, Exception):
            raise turn
        return turn  # type: ignore[return-value]


def encode_tool_call(call: ToolCall) -> str:
    return json.dumps(
        {
            "tool_call": {
                "id": call.call_id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def encode_tool_success(result: object) -> str:
    return json.dumps(
        {"ok": True, "result": result},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def encode_tool_error(exc: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        },
        ensure_ascii=False,
        sort_keys=True,
    )

