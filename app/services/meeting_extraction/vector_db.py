
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector

from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict


def get_embeddings(model_name: str = "BAAI/bge-m3", device: Literal["cpu", "cuda"] = "cpu"):
    """임베딩 모델 인스턴스를 생성합니다."""

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_neo4j_vector_store(
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    collection_name: str,
    node_label: str = "Decision",
    emb_model_name: str = "BAAI/bge-m3",
    emb_device: Literal["cpu", "cuda"] = "cpu",
) -> Neo4jVector:
    """
    Neo4jVector 인스턴스를 생성

    Args:
        - neo4j_url (str) : neo4j 주소
        - neo4j_user (str) : neo4j 유저이름
        - neo4j_password (str) : neo4j 암호
        - collection_name: (str) : 
        - node_label (str) : 노드 분류, default="Decision"
        - emb_model_name (str) : 임베딩 모델 이름, default="BAAI/bge-m3"
        - emb_model_device (str) : 임베딩 디바이스 위치, cpu or cuda, default="cpu"

    """
    # 검색 결과의 metadata에서 text/embedding 본문 필드는 제외합니다.
    retrieval_query = """
    RETURN node.`text` AS text, score,
           node {.*, `text`: Null, `embedding`: Null} AS metadata
    """

    store = Neo4jVector(
        embedding=get_embeddings(emb_model_name, emb_device),
        url=neo4j_url,
        username=neo4j_user,
        password=neo4j_password,
        index_name=f"{collection_name}_vector",
        node_label=node_label,
        text_node_property="text",
        embedding_node_property="embedding",
        retrieval_query=retrieval_query,
    )

    # 벡터 인덱스가 없다면 생성 
    info = store.retrieve_existing_index()
    if info is None or info[0] is None:
        store.create_new_index()
    return store
