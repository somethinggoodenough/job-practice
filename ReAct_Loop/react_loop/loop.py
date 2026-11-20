#Each time we get a message, we decide what to do next here
# messages should look like this:
# messages = [
#     {"role": "system", "content": xxx},
#     {"role": "system", "content": xxx},
#     {"role": "system", "content": xxx},
#     ]
def llm(messages: list[dict]) -> dict:
    



tools: dict[str, Callable[..., str]]

#we do the looping here
def run_agent(task: str, tools, max_steps: int = 10) -> str:
    max_step = 100
    for step in range(max_step):
