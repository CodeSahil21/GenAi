from langchain_community.document_loaders import PyPDFLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(documents=docs)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# vector_store  = QdrantVectorStore.from_documents(
#     documents=[],
#     url="http://localhost:6333",
#     collection_name="nodejs",
#     embedding=embeddings
# )

# vector_store.add_documents(documents=split_docs)
# print("Ingestion done")
# print("Docs count:", len(docs))
# print("Split docs count:", len(split_docs))
# print

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embeddings
)

relevant_chunks  = retriever.similarity_search(
    query="What is FS Module?"
)

SYSTEM_PROMPT = """ You are a helpful assistant  who responds base of the available context

context:
{relevant_chunks}
"""