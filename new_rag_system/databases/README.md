# Dual Vector Databases

## Overview

This RAG-heavy system uses TWO specialized databases for better retrieval:

1. **Text Database** - Pattern and description matching
2. **Code Database** - Code structure similarity

## Database Details

### Text Database (text_db/)
- **Model**: BAAI/bge-large-en-v1.5
- **Embedding Dimension**: 1024
- **Content**: Full vulnerability descriptions and explanations
- **Purpose**: Match patterns, keywords, vulnerability descriptions
- **Chunk Size**: 1000 chars (200 overlap)

### Code Database (code_db/)
- **Model**: microsoft/graphcodebert-base
- **Embedding Dimension**: 768
- **Content**: Code snippets extracted from findings
- **Purpose**: Find structurally similar vulnerable code
- **Granularity**: Individual code blocks

## Data Source

- **Location**: `../../sample-smart-contract-dataset/`
- **Files**: 912 curated audit findings (JSON format)
- **Quality**: Professional security audit reports
- **Firms**: Quantstamp, Trail of Bits, Code4rena, Cyfrin, etc.

## Building Databases

```bash
cd new_rag_system/databases/
python build_dual_databases.py
```

**Time**: 10-20 minutes total
**Output**:
- `text_db/` - Text embeddings
- `code_db/` - Code embeddings

## Why Dual Databases?

Single database approach (like simplified system):
- Uses one embedding model for both code and text
- Less specialized retrieval

Dual database approach (this system):
- GraphCodeBERT understands code structure semantics
- BGE-Large excels at natural language pattern matching
- Better retrieval quality through specialization
- More sophisticated but more effective

## Database Size

Expected sizes after building:
- Text DB: ~50-80 MB
- Code DB: ~30-50 MB
- Total: ~80-130 MB

Much smaller than old system (189MB + 68MB) because:
- Cleaner baseline dataset (912 vs 8,358 findings)
- Focus on quality over quantity
