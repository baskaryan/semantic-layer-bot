import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.indexing import index, InMemoryRecordManager
import yaml

CUR_DIR = Path(__file__).parent


def main() -> None:
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    # TODO: Replace with non-local vectorstore.
    vecstore = Chroma("dbt_demo", embedding_function=embedding, persist_directory=".")

    # TODO: Using in memory record manager for demo purposes, replace with persistent
    # manager.
    record_manager = InMemoryRecordManager(namespace="dbt_demo")
    docs = chunk_manifest(CUR_DIR / "semantic_manifest.yaml")
    res = index(docs, record_manager, vecstore)
    print(res)


"""
Questions for improving index/retrieval:
    - Do we need to chunk and retrieve at all? How big is the typical manifest, could 
        it just all be in context?
    - If the whole manifest doesn't fit, could just the names of all models/entities/
        dimensions/metrics fit in context? And then model can choose to look up the 
        definitions of the most relevant objects?
    - Assuming retrieval is needed, probably would get better results through hybrid 
        search that combines keyword and semantic search.
    - What's the right information to be in each vector? Could it just be the name and 
        description of the object? Does it help to include parent descriptions (eg the 
        semantic model description)?
    - Would parent-document retrieval help?
    - Would query analysis, with things like query decomposition, help?
"""


def chunk_manifest(path):
    with open(path, "r") as f:
        manifest = yaml.safe_load(f)
    docs = []
    categories = ("entities", "dimensions", "measures")
    for model in manifest["semantic_models"]:
        for category in categories:
            for val in model.get(category, []):
                docs.append(
                    Document(
                        json.dumps(val, indent=2),
                        metadata={"category": category},
                    )
                )
    for metric in manifest["metrics"]:
        docs.append(
            Document(
                json.dumps(metric, indent=2),
                metadata={"category": "metrics"},
            )
        )
    return docs


if __name__ == "__main__":
    main()
