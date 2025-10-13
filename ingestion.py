import os

import nest_asyncio
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_share_data, initialize_pipeline_status
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
# allow nested loops (Jupyter or other already‐running loops)
nest_asyncio.apply()


load_dotenv()



async def initialize_rag(working_dir: str = "./rag_storage") -> LightRAG:
    """
    Initialize LightRAG with vector and Neo4j graph storage,
    and prepare shared pipeline status to avoid KeyError.
    """
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2:latest"),
        summary_max_tokens=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "options": {"num_ctx": 32768},
            "timeout": int(os.getenv("TIMEOUT", "300000")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )
    await rag.initialize_storages()

    # ensure shared dicts exist
    #initialize_share_data()
    await initialize_pipeline_status()

    return rag

async def index_data(rag: LightRAG, file_path: str) -> None:
    """
    Index a text file into LightRAG, tagging chunks with its filename.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # stream chunks into vector store and graph
    await rag.ainsert(input=text, file_paths=[file_path])

async def index_file(rag: LightRAG, path: str) -> None:
    """
    Alias for index_data to mirror sync naming.
    """
    await index_data(rag, path)

