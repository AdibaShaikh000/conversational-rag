from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def build_vector_store(chunks: list[str], api_key: str) -> FAISS:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_texts(chunks, embeddings)


def get_base_retriever(vector_store: FAISS, k: int = 20):
    return vector_store.as_retriever(search_kwargs={"k": k})
