import io

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "".join(page.extract_text() or "" for page in reader.pages)


def load_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "".join(page.extract_text() or "" for page in reader.pages)


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)
