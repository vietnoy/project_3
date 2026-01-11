#!/usr/bin/env python3
"""
Build Dual Vector Databases for RAG-Heavy System
================================================
Creates two separate databases:
1. Code Database (GraphCodeBERT) - for code similarity
2. Text Database (BGE-Large) - for pattern/description matching

Data source: sample-smart-contract-dataset/ (912 findings)
"""
import json
import glob
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


def extract_code_blocks(content: str) -> List[str]:
    """Extract code blocks from markdown content"""
    # Match ```solidity, ```javascript, or just ``` blocks
    pattern = r'```(?:solidity|javascript|typescript|sol|js|ts)?\s*\n(.*?)```'
    code_blocks = re.findall(pattern, content, re.DOTALL)

    # Filter out very short snippets
    return [code.strip() for code in code_blocks if len(code.strip()) > 50]


def build_text_database(
    data_dir: str = '../../sample-smart-contract-dataset',
    persist_dir: str = './text_db'
):
    """
    Build text database with BGE-Large embeddings

    Contains: Full vulnerability descriptions for pattern matching
    """
    print("=" * 80)
    print("BUILDING TEXT DATABASE")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Output: {persist_dir}")
    print("Model: BAAI/bge-large-en-v1.5")
    print("=" * 80)
    print()

    # Load JSON files
    json_files = glob.glob(f"{data_dir}/*.json")
    print(f"Found {len(json_files)} audit findings")
    print()

    # Create documents
    documents = []
    print("Loading findings...")

    for filepath in tqdm(json_files, desc="Loading", unit="files"):
        try:
            with open(filepath, 'r') as f:
                finding = json.load(f)

            finding_id = Path(filepath).stem
            content = finding.get('content', '')

            if not content:
                continue

            metadata = {
                'finding_id': finding_id,
                'source': filepath,
                'title': finding.get('title', 'N/A'),
                'firm_name': finding.get('firm_name', 'N/A'),
                'protocol_name': finding.get('protocol_name', 'N/A'),
                'impact': finding.get('impact', 'MEDIUM'),
                'reference_link': finding.get('reference_link', 'N/A'),
                'report_date': finding.get('report_date', 'N/A'),
                'doc_type': 'text'
            }

            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))

        except Exception as e:
            print(f"Warning: Error loading {filepath}: {e}")
            continue

    print(f"Loaded {len(documents)} documents")
    print()

    # Split into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks for better retrieval
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")
    print()

    # Build database
    print("Building vector database...")
    print("Loading BGE-Large embedding model (this may take a moment)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    print("Creating empty vectorstore...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Add documents in batches to avoid hanging
    # Small batch size for consistent performance
    batch_size = 25
    print(f"Adding documents in batches of {batch_size}...")
    print(f"Total batches: {(len(splits) + batch_size - 1) // batch_size}")

    for i in tqdm(range(0, len(splits), batch_size), desc="Embedding", unit="batch"):
        batch = splits[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("All documents added successfully")

    print()
    print("=" * 80)
    print("TEXT DATABASE COMPLETE")
    print("=" * 80)
    print(f"Location: {persist_dir}/")
    print(f"Total embeddings: {len(splits)}")
    print(f"Embedding dimension: 1024")
    print("=" * 80)
    print()

    return vectorstore


def build_code_database(
    data_dir: str = '../../sample-smart-contract-dataset',
    persist_dir: str = './code_db'
):
    """
    Build code database with GraphCodeBERT embeddings

    Contains: Code snippets for code similarity search
    """
    print("=" * 80)
    print("BUILDING CODE DATABASE")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Output: {persist_dir}")
    print("Model: microsoft/graphcodebert-base")
    print("=" * 80)
    print()

    # Load JSON files and extract code
    json_files = glob.glob(f"{data_dir}/*.json")
    print(f"Found {len(json_files)} audit findings")
    print()

    code_documents = []
    total_code_blocks = 0

    print("Extracting code blocks...")

    for filepath in tqdm(json_files, desc="Extracting", unit="files"):
        try:
            with open(filepath, 'r') as f:
                finding = json.load(f)

            finding_id = Path(filepath).stem
            content = finding.get('content', '')

            # Extract code blocks
            code_blocks = extract_code_blocks(content)
            total_code_blocks += len(code_blocks)

            # Create document for each code block
            for idx, code in enumerate(code_blocks):
                metadata = {
                    'finding_id': finding_id,
                    'source': filepath,
                    'title': finding.get('title', 'N/A'),
                    'firm_name': finding.get('firm_name', 'N/A'),
                    'protocol_name': finding.get('protocol_name', 'N/A'),
                    'impact': finding.get('impact', 'MEDIUM'),
                    'reference_link': finding.get('reference_link', 'N/A'),
                    'snippet_index': idx,
                    'code_lines': len(code.split('\n')),
                    'doc_type': 'code'
                }

                code_documents.append(Document(
                    page_content=code,
                    metadata=metadata
                ))

        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue

    print(f"Extracted {total_code_blocks} code blocks from {len(json_files)} findings")
    print(f"Created {len(code_documents)} code documents")
    print()

    if len(code_documents) == 0:
        print("ERROR: No code blocks found!")
        return None

    # Build database
    print("Building vector database...")
    print("Loading GraphCodeBERT embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="microsoft/graphcodebert-base"
    )

    print("Creating empty vectorstore...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Add documents in batches to avoid hanging
    # Small batch size for consistent performance
    batch_size = 25
    print(f"Adding code documents in batches of {batch_size}...")
    print(f"Total batches: {(len(code_documents) + batch_size - 1) // batch_size}")

    for i in tqdm(range(0, len(code_documents), batch_size), desc="Embedding", unit="batch"):
        batch = code_documents[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("All code documents added successfully")

    print()
    print("=" * 80)
    print("CODE DATABASE COMPLETE")
    print("=" * 80)
    print(f"Location: {persist_dir}/")
    print(f"Total code embeddings: {len(code_documents)}")
    print(f"Embedding dimension: 768")
    print("=" * 80)
    print()

    return vectorstore


def main():
    """Build both databases"""
    print()
    print("=" * 80)
    print("DUAL DATABASE BUILDER FOR RAG-HEAVY SYSTEM")
    print("=" * 80)
    print("Building two specialized databases:")
    print("  1. Text DB (BGE-Large) - for pattern/description matching")
    print("  2. Code DB (GraphCodeBERT) - for code similarity search")
    print()
    print("Data source: 912 curated audit findings")
    print("Estimated time: 10-20 minutes total")
    print("=" * 80)
    print()

    # Build text database
    print("Starting text database build...")
    print()
    text_db = build_text_database()

    print()
    print("Text database complete. Building code database...")
    print()

    # Build code database
    code_db = build_code_database()

    # Summary
    print()
    print("=" * 80)
    print("ALL DATABASES BUILT SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Your RAG-heavy system now has:")
    print("  [TEXT] ./new_rag_system/databases/text_db/")
    print("  [CODE] ./new_rag_system/databases/code_db/")
    print()
    print("Next steps:")
    print("  1. Build structural extractor")
    print("  2. Build RAG retriever")
    print("  3. Build RAG detector")
    print("  4. Run evaluation")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()
