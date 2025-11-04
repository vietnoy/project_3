"""
Vector database setup and management

This module handles:
- Loading JSON vulnerability findings
- Creating embeddings
- Storing in ChromaDB
- Providing retrieval interface
"""

from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os


def load_vulnerability_database(data_dir: str = "sample-smart-contract-dataset",
                                persist_dir: str = "./chroma_db"):
    """
    Load vulnerability findings into vector database

    Args:
        data_dir: Directory containing JSON files
        persist_dir: Where to save ChromaDB

    Returns:
        ChromaDB vectorstore instance
    """

    # Check if database already exists
    if os.path.exists(persist_dir):
        print(f"Loading existing database from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings()
        )

    print("Creating new database...")

    # Load JSON files
    # Note: You may need to adjust jq_schema based on your JSON structure
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

    # Split long documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    print("Splitting documents...")
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    # Create embeddings and store in ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )

    print(f"Database created and saved to {persist_dir}")
    return vectorstore


def search_similar_vulnerabilities(vectorstore, query: str, k: int = 5):
    """
    Search for similar vulnerabilities

    Args:
        vectorstore: ChromaDB instance
        query: Search query
        k: Number of results to return

    Returns:
        List of similar documents
    """
    return vectorstore.similarity_search(query, k=k)


if __name__ == "__main__":
    # Test the database loading
    db = load_vulnerability_database()

    # Test search
    results = search_similar_vulnerabilities(db, "reentrancy attack")
    print(f"\nTest search results: {len(results)} findings")
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.page_content[:100]}...")
