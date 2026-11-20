from __future__ import annotations

import importlib
import json
import unittest

from react_loop.contracts import FinalAnswer, Message, RunResult, ToolCall
from react_loop.environment import ScriptedModel, ToolRegistry


def load_runner():
    module = importlib.import_module("react_loop.loop")
    runner = getattr(module, "run_react_loop", None)
    if runner is None:
        raise AssertionError(
            "请先在 react_loop/loop.py 中实现 run_react_loop"
        )
    return runner


class ReActLoopTests(unittest.TestCase):
    def test_returns_a_direct_final_answer(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel([FinalAnswer("Detroit")])

        result = run_react_loop(model, ToolRegistry(), "Where?", max_steps=3)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "Detroit")
        self.assertEqual(result.steps, 1)
        self.assertEqual(result.error, None)
        self.assertEqual(
            result.messages,
            (
                Message(role="user", content="Where?"),
                Message(role="assistant", content="Detroit"),
            ),
        )

    def test_dispatches_tool_and_returns_observation_to_model(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel(
            [
                ToolCall("call-1", "add", {"left": 2, "right": 3}),
                FinalAnswer("The answer is 5."),
            ]
        )
        tools = ToolRegistry()
        tools.register("add", lambda left, right: left + right)

        result = run_react_loop(model, tools, "Add 2 and 3")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, 2)
        self.assertEqual([call.name for call in tools.calls], ["add"])
        observation = result.messages[2]
        self.assertEqual(observation.role, "tool")
        self.assertEqual(observation.name, "add")
        self.assertEqual(observation.call_id, "call-1")
        self.assertEqual(json.loads(observation.content), {"ok": True, "result": 5})
        self.assertEqual(model.histories[1], result.messages[:3])

    def test_unknown_tool_becomes_observation_and_loop_continues(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel(
            [ToolCall("bad-1", "missing", {}), FinalAnswer("I recovered.")]
        )

        result = run_react_loop(model, ToolRegistry(), "Try a tool")

        self.assertEqual(result.status, "completed")
        error = json.loads(result.messages[2].content)
        self.assertFalse(error["ok"])
        self.assertEqual(error["error"]["type"], "UnknownToolError")
        self.assertIn("unknown tool", error["error"]["message"])

    def test_tool_exception_becomes_observation_and_loop_continues(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel(
            [ToolCall("boom-1", "explode", {}), FinalAnswer("Used a fallback.")]
        )
        tools = ToolRegistry()

        def explode() -> None:
            raise OSError("disk unavailable")

        tools.register("explode", explode)
        result = run_react_loop(model, tools, "Do the work")

        self.assertEqual(result.status, "completed")
        error = json.loads(result.messages[2].content)
        self.assertEqual(error["error"]["type"], "ToolExecutionError")
        self.assertIn("disk unavailable", error["error"]["message"])

    def test_stops_at_max_steps(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel(
            [
                ToolCall("one", "ping", {}),
                ToolCall("two", "ping", {}),
                FinalAnswer("This turn must not run"),
            ]
        )
        tools = ToolRegistry()
        tools.register("ping", lambda: "pong")

        result = run_react_loop(model, tools, "Keep going", max_steps=2)

        self.assertEqual(result.status, "max_steps")
        self.assertEqual(result.answer, None)
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.error, "maximum steps reached")
        self.assertEqual(len(model.histories), 2)

    def test_model_failure_returns_model_error(self) -> None:
        run_react_loop = load_runner()
        model = ScriptedModel([ConnectionError("provider unavailable")])

        result = run_react_loop(model, ToolRegistry(), "Hello")

        self.assertEqual(result.status, "model_error")
        self.assertEqual(result.answer, None)
        self.assertEqual(result.steps, 1)
        self.assertIn("ConnectionError", result.error)
        self.assertIn("provider unavailable", result.error)

    def test_rejects_invalid_step_limit_and_invalid_model_turn(self) -> None:
        run_react_loop = load_runner()
        with self.assertRaises(ValueError):
            run_react_loop(ScriptedModel([]), ToolRegistry(), "Hello", max_steps=0)

        result: RunResult = run_react_loop(
            ScriptedModel([{"unexpected": True}]), ToolRegistry(), "Hello"
        )
        self.assertEqual(result.status, "model_error")
        self.assertEqual(result.steps, 1)
        self.assertIn("unexpected model turn", result.error)


if __name__ == "__main__":
    unittest.main()
