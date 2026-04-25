from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_reranking_retriever(
    base_retriever,
    model_name: str = RERANKER_MODEL,
    top_n: int = 5,
) -> ContextualCompressionRetriever:
    """
    Wraps base_retriever with a cross-encoder reranker.
    Fetches k candidates (set on base_retriever), reranks, returns top_n.
    """
    encoder = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=encoder, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
