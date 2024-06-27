# semantic layer bot

## Setup

```bash
poetry install --with dev
```

## To interact with real semantic layer
- Replace semantic_manifest.yaml with real manifest
- Remove the mocked dbt calls in agent.py file (`_get_dbt_query_result()` and `_create_dbt_query()`)

## Run indexing
```bash
poetry run python ingest.py
```

## Use agent
```python
from agent import graph

async for event in graph.astream_events({"messages": [("human", "how much was ordered in total in july")]}, version="v2", include_types=["chat_model", "tool"]):
    print(event['event'])
    print(event['data'])
    print("\n\n" + "-" * 80 + "\n\n")
```

## TODO
- Is retrieval even needed? If so:
    - Use a real vecstore (not a local one)
    - Use a real record manager (not an in mem one)
    - Do less naive retrieval (hybrid semantc + keyword search, include parent information in each chunk, etc)
- Improve prompts, add few-shot examples
- Make sure dbt api calls are correct 


## LangGraph docs

https://langchain-ai.github.io/langgraph/
