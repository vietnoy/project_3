"""
Basic tests for Smart Contract Security Assistant
"""

import pytest
from src.database import load_vulnerability_database, search_similar_vulnerabilities


def test_database_loading():
    """Test that database loads successfully"""
    # This will take time on first run (creating embeddings)
    db = load_vulnerability_database()
    assert db is not None


def test_search_functionality():
    """Test basic search"""
    db = load_vulnerability_database()
    results = search_similar_vulnerabilities(db, "reentrancy", k=3)
    assert len(results) > 0
    assert len(results) <= 3


# Add more tests as you develop features
