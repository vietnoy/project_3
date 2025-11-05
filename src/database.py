"""Vector database setup using ChromaDB and HuggingFace embeddings"""

from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


def load_vulnerability_database(data_dir: str = "sample-smart-contract-dataset",
                                persist_dir: str = "./chroma_db"):
    """Load vulnerability findings into ChromaDB vector database"""

    # Check if database already exists
    if os.path.exists(persist_dir):
        print(f"Loading existing database from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

    print("Creating new database...")

    loader = DirectoryLoader(
        data_dir,
        glob="**/*.json",
        loader_cls=JSONLoader,
        loader_kwargs={
            'jq_schema': '.',  # Load entire JSON
            'text_content': False
        }
    )

    print("Loading documents...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    print("Splitting documents...")
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=persist_dir
    )

    print(f"Database created and saved to {persist_dir}")
    return vectorstore


def search_similar_vulnerabilities(vectorstore, query: str, k: int = 5):
    """Search for similar vulnerabilities in the vector database"""
    return vectorstore.similarity_search(query, k=k)
