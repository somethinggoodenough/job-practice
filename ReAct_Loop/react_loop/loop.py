def llm(messages: list[dict]) -> dict:




tools: dict[str, Callable[..., str]]

def run_agent(task: str, tools, max_steps: int = 10) -> str:
    