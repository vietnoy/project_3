# New RAG-Heavy Vulnerability Detection System

## Philosophy

TRUE RAG system where:
- Feature extraction is MINIMAL (structural patterns only, no vulnerability names)
- Intelligence comes from EMBEDDINGS and RETRIEVAL (70%)
- LLM DISCOVERS vulnerabilities by comparing to similar examples
- Proper evaluation of LLM output (not just feature extraction)

## Folder Structure

```
new_rag_system/
├── databases/
│   ├── build_dual_databases.py     # Build GraphCodeBERT + BGE databases
│   └── README.md                    # Database documentation
├── evaluation/
│   └── run_evaluation.py            # Test system with LLM metrics
├── tests/
│   └── test_rag_detector.py         # Unit tests
├── structural_extractor.py          # Minimal feature extraction
├── rag_retriever.py                 # Heavy retrieval logic
├── rag_detector.py                  # Main RAG detector
└── README.md                        # This file
```

## Key Differences from Old System

| Component | Old System (Feature-Heavy) | New System (RAG-Heavy) |
|-----------|---------------------------|------------------------|
| Feature Extraction | Returns specific vulnerability names | Returns only structural patterns |
| Retrieval | 30% of intelligence | 70% of intelligence |
| LLM Role | Cites pre-detected vulnerabilities | Discovers vulnerabilities from examples |
| Evaluation | Tests feature extraction only | Tests LLM output quality |

## Data Source

- Baseline: 912 curated findings from `sample-smart-contract-dataset/`
- Quality over quantity - build solid foundation first
- Expand later once system works well

## No Emojis

All code is clean, professional, no emoji icons.

## Next Steps

1. Build dual databases (GraphCodeBERT + BGE)
2. Implement structural extractor
3. Implement RAG retriever
4. Implement RAG detector
5. Evaluate with proper LLM metrics
