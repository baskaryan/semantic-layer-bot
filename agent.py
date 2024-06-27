import json
import os
import time
from collections import defaultdict
from typing import TypedDict, Annotated, Sequence, Optional, List, Union, Literal, Any

from functools import lru_cache

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, add_messages
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class GraphConfig(TypedDict):
    model: str


# TODO: Replace with your vectorstore.
vecstore = Chroma(
    "dbt_demo",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=".",
)


class QueryDataSchemaInput(BaseModel):
    """Get information about the schema of the data via a natural language query."""

    query: str


@tool(args_schema=QueryDataSchemaInput)
def query_data_schema(query: str) -> str:
    """Get information about the schema of the data via a natural language query."""
    docs = vecstore.similarity_search(query, k=5)
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc.metadata["category"]].append(json.loads(doc.page_content))
    return json.dumps(grouped, indent=2)


class GroupBy(BaseModel):
    name: str
    grain: Optional[Literal["DAY", "WEEK", "MONTH", "QUARTER", "YEAR"]] = None


class OrderBy(BaseModel):
    by: Union[str, GroupBy]
    descending: bool = False


METRICS_DESCRIPTION = """
The metrics to return.

For any time-series metric, the 'metric_time' keyword should always be available for use in queries.

Example: If you had a 'revenue' metric, you could query for `metrics=['revenue']`
"""

GROUP_BY_DESCRIPTION = """
Dimension names or entities to group by. We require a reference to the entity of the dimension (other than for the primary time dimension 'metric_time'), which is pre-appended to the front of the dimension name with a double underscore.

Example: If you had a 'user' entity and 'country' dimension, you could group by `group_by=['user__country', 'metric_time']`
"""

WHERE_DESCRIPTION = """
The where filter takes a list of strings that represent SQL WHERE conditions. Depending on the object you are filtering, there are a couple of parameters:

Dimension() — Used for any categorical or time dimensions. For example, `where=["{{ Dimension('customer__country') }} = 'US'", "{{ Dimension('metric_time').grain('month') }} > '2022-10-01'"]`.
Entity() — Used for entities like primary and foreign keys, such as `where=["{{ Entity('order_id') > 10 }}"].
Note: If you prefer a where clause with a more explicit path, you can optionally use TimeDimension() to separate categorical dimensions from time ones. The TimeDimension input takes the time dimension and optionally the granularity level. TimeDimension('metric_time', 'month').
"""

ORDER_DESCRIPTION = """
Order the data returned by a particular field.
"""


class QueryDataInput(BaseModel):
    """Get a relevant view of the requested metrics via a structured query."""

    metrics: List[str] = Field(..., description=METRICS_DESCRIPTION)
    group_by: Optional[List[GroupBy]] = Field(None, description=GROUP_BY_DESCRIPTION)
    where: Optional[List[str]] = Field(None, description=WHERE_DESCRIPTION)
    order: Optional[List[OrderBy]] = Field(None, description=ORDER_DESCRIPTION)
    limit: Optional[int] = Field(None, description="Limit number of results returned.")


@tool(args_schema=QueryDataInput)
def query_data(
    metrics: List[str],
    group_by: Optional[List[GroupBy]] = None,
    where: Optional[List[str]] = None,
    order: Optional[List[OrderBy]] = None,
    limit: Optional[int] = None,
) -> dict:
    """Get a relevant view of the requested metrics via a structured query."""
    api_token = os.environ["DBT_API_TOKEN"]
    env_id = os.environ.get("DBT_ENV_ID", 0)
    headers = {"Authorization": f"Bearer {api_token}"}

    query_lines = [
        f"environmentId: {env_id}",
        "metrics: [" + ", ".join(_metric_to_str(metric) for metric in metrics) + "]",
    ]
    if group_by:
        query_lines.append(
            "groupBy: [" + ", ".join(_group_by_to_str(by) for by in group_by) + "]"
        )
    if where:
        where = [w.replace('"', "'") for w in where]
        query_lines.append(
            "where: [" + ", ".join(f'{{sql: "{w}"}}' for w in where) + "]"
        )
    if order:
        query_lines.append(
            "orderBy: [" + ", ".join(_order_by_to_str(by) for by in order) + "]"
        )
    if limit:
        query_lines.append(f"limit: {limit}")

    query = "\n\t".join(query_lines)
    query_id = _create_dbt_query(query, headers)
    return _get_dbt_query_result(query_id, env_id, headers)


def _metric_to_str(metric: str) -> str:
    return f'{{name: "{metric}"}}'


def _order_by_to_str(order_by: OrderBy) -> str:
    if isinstance(order_by.by, str):
        str_ = _metric_to_str(order_by.by)
    else:
        str_ = _group_by_to_str(order_by.by)
    if order_by.descending:
        str_ += f", descending: {str(order_by.descending).lower()}"
    return "{" + str_ + "}"


def _group_by_to_str(by: OrderBy) -> str:
    if by.grain:
        return f'{{name: "{by.name}", grain: {by.grain}}}'
    else:
        return f'{{name: "{by.name}"}}'


def _create_dbt_query(query: str, headers: dict) -> str:
    # TODO: Delete mock return
    return "123"

    create_query_request = f"""
mutation {{
  createQuery(
    {query}
  ) {{
    queryId
  }}
}}
    """
    gql_response = requests.post(
        "https://semantic-layer.cloud.getdbt.com/api/graphql",
        json={"query": create_query_request},
        headers=headers,
    )
    return gql_response.json()["data"]["queryId"]


def _get_dbt_query_result(query_id: str, env_id: str, headers: dict) -> dict:
    # TODO: Delete mock return
    return {
        "sql": "SELECT\n  ordered_at AS metric_time__day\n  , SUM(order_total) AS order_total\nFROM semantic_layer.orders orders_src_1\nGROUP BY\n  ordered_at",
        "jsonResult": {
            "order_total": [1, 10, 123],
            "ordered_at": ["2023-07-01", "2023-07-02", "2023-07-03"],
        },
    }

    query_result_request = f"""
{{
  query(environmentId: {env_id}, queryId: {query_id}) {{
    sql
    status
    jsonResult
  }}
}}
"""
    while True:
        gql_response = requests.post(
            "https://semantic-layer.cloud.getdbt.com/api/graphql",
            json={"query": query_result_request},
            headers=headers,
        )
        response_json = gql_response.json()["data"]
        if response_json.pop("status") in ["FAILED", "SUCCESSFUL"]:
            break
        # Set an appropriate interval between polling requests
        time.sleep(1)
    resposne_json["jsonResult"] = json.loads(response_json["jsonResult"])
    return response_json


@lru_cache(maxsize=4)
def _cached_model(model_name: str, **kwargs: Any) -> BaseChatModel:
    return init_chat_model(model_name, **kwargs)


def continue_(state):
    return "tools" if state["messages"][-1].tool_calls else END


DEFAULT_MODEL = "claude-3-5-sonnet-20240620"


metadata_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant who is very good at analyzing data. Before you \
begin writing any data queries, you first need to understand what fields you can \
query. To start, use the query_data_schema tool to get the definitions of all the 
relevant fields in your db.""",
        ),
        # TODO: Add few-shot examples
        ("placeholder", "{messages}"),
    ]
)


def write_metadata_query(state, config):
    config = config.get("configurable", {})
    model_name = config.get(
        "metadata_model",
    ) or config.get("model", DEFAULT_MODEL)
    model = _cached_model(model_name).bind_tools([query_data_schema], tool_choice="any")
    chain = metadata_prompt | model
    return {"messages": [chain.invoke({"messages": state["messages"]})]}


query_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant who is very good at analyzing data. You have \
access to a user's data via the DBT Semantic Layer API. Use the query_data tool as \
needed to get the data requested by the user. If you already have the information \
needed to respond to the user, you should respond directly. If the data the user \
is requesting is not available, you should say so.

Make sure to only query metrics, entities, and dimensions that you are sure exist.""",
        ),
        # TODO: Add few-shot examples.
        ("placeholder", "{messages}"),
    ]
)


def query_or_answer(state, config):
    config = config.get("configurable", {})
    model_name = config.get(
        "query_model",
    ) or config.get("model", DEFAULT_MODEL)
    model = _cached_model(model_name).bind_tools([query_data])
    chain = query_prompt | model
    return {"messages": [chain.invoke({"messages": state["messages"]})]}


workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node(write_metadata_query)
workflow.add_node(query_or_answer)
workflow.add_node(ToolNode([query_data_schema, query_data]))

workflow.set_entry_point("write_metadata_query")
workflow.add_edge("write_metadata_query", "tools")
workflow.add_edge("tools", "query_or_answer")
workflow.add_conditional_edges("query_or_answer", continue_)

graph = workflow.compile()
