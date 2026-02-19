from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Any, Dict, List, Optional, Literal

def get_embeddings(emb_model_name: str = "BAAI/bge-m3", device: Literal["cpu", "cuda"] = "cpu"):
    """
    embeddings 생성
    """
    return HuggingFaceEmbeddings(
        model_name=emb_model_name,
        model_kwargs={"device": device}, # CUDA면 "cuda"
        encode_kwargs={"normalize_embeddings": True}
    )

