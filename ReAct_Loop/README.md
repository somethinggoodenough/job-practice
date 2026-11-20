# ReAct Agent Loop 练习

这个练习聚焦三个面试常考点：

- tool dispatch：根据模型返回的工具名和参数调用正确工具；
- 停止条件：模型给出最终答案，或循环达到 `max_steps`；
- 错误处理：未知工具、工具执行异常、模型异常都不能让循环失控。

项目只使用 Python 标准库，不需要安装第三方依赖。建议使用 Python 3.11+。

## 你要实现的文件

`react_loop/loop.py` 已创建并保持为空。请在其中实现：

```python
def run_react_loop(
    model: AgentModel,
    tools: ToolRegistry,
    prompt: str,
    *,
    max_steps: int = 5,
) -> RunResult:
    ...
```

可直接使用 `react_loop.contracts` 和 `react_loop.environment` 中已经提供的类型与执行环境。

## 行为约定

1. `max_steps <= 0` 时抛出 `ValueError`。
2. 消息历史以 `Message(role="user", content=prompt)` 开始。
3. 每一步调用一次 `model.next_turn(messages)`。
4. 模型返回 `ToolCall` 时：
   - 先向历史加入一条 `assistant` 消息，内容由 `encode_tool_call` 生成；
   - 调用 `tools.invoke(turn)`；
   - 成功时加入由 `encode_tool_success` 生成的 `tool` 消息；
   - 捕获 `UnknownToolError` 或 `ToolExecutionError`，加入由 `encode_tool_error` 生成的 `tool` 消息，然后继续循环。
5. 模型返回 `FinalAnswer` 时，加入一条 `assistant` 消息并返回：
   - `status="completed"`
   - `answer=turn.content`
   - `steps` 为已经调用模型的次数。
6. 达到步数上限仍没有最终答案时返回：
   - `status="max_steps"`
   - `answer=None`
   - `error="maximum steps reached"`
7. 模型抛出异常，或返回 `ToolCall` / `FinalAnswer` 以外的值时，不继续循环，返回：
   - `status="model_error"`
   - `answer=None`
   - `error` 为可读的错误信息。
8. 传给模型的消息历史应为只读快照，避免模型意外修改 loop 内部状态。

`RunResult.messages` 保存完整消息历史。构造结果时建议把消息列表转换为 tuple。

## 运行测试

在当前目录执行：

```bash
python -m unittest discover -s tests -v
```

初始状态下测试失败是正常的，因为 `react_loop/loop.py` 是空文件。实现完成后应有 7 个测试通过。

## 文件说明

```text
ReAct_Loop/
├── README.md
├── pyproject.toml
├── react_loop/
│   ├── __init__.py
│   ├── contracts.py      # 消息、模型输出和运行结果类型
│   ├── environment.py    # 工具注册器、假模型和消息编码函数
│   └── loop.py           # 由你实现（刻意留空）
└── tests/
    └── test_loop.py      # 验收测试
```

