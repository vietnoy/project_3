# RAG-Heavy Smart Contract Vulnerability Detection System

A data-driven approach using Retrieval-Augmented Generation (RAG) to detect vulnerabilities in smart contracts.

## 📊 Performance

- **F1 Score**: 80%
- **Precision**: 80%
- **Recall**: 90%
- **Database**: 912 professional audit findings
- **Architecture**: 30% Structural Extraction, 40% RAG Retrieval, 30% LLM Discovery

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Ollama & Model

```bash
# Download from https://ollama.com
# Or use brew (macOS)
brew install ollama

# Pull Qwen2.5-Coder model
ollama pull qwen2.5-coder:7b
```

### 3. Build Databases (REQUIRED)

**Note**: Databases are NOT included in git (too large). You must build them locally:

```bash
cd new_rag_system/databases
../../venv/bin/python build_dual_databases.py
```

**Time**: 10-20 minutes
**Output**: `text_db/` (247MB) and `code_db/` (93MB)

### 4. Verify Setup

```bash
venv/bin/python -c "from new_rag_system.rag_detector import RAGVulnerabilityDetector; \
                     detector = RAGVulnerabilityDetector(); \
                     print('✓ Setup Complete!')"
```

## 🧪 Run Evaluation

```bash
cd new_rag_system/evaluation
../../venv/bin/python run_evaluation.py
```

**Expected Results:**
- Tests Passed: 8/10 (80%)
- Precision: 80%
- Recall: 90%
- F1 Score: 80%

## 💻 Basic Usage

```python
from new_rag_system.rag_detector import RAGVulnerabilityDetector

# Initialize detector
detector = RAGVulnerabilityDetector()

# Analyze code
code = """
function withdraw() public {
    uint256 amount = balances[msg.sender];
    msg.sender.call{value: amount}("");
    balances[msg.sender] = 0;
}
"""

result = detector.detect(code, verbose=True)
print(result['analysis'])
```

## 📁 Project Structure

```
project_3/
├── new_rag_system/                   # Main RAG system
│   ├── rag_detector.py               # Main detector
│   ├── rag_retriever.py              # Dual database retrieval
│   ├── structural_extractor.py       # Pattern extraction
│   ├── evaluation/
│   │   ├── run_evaluation.py         # Evaluation script
│   │   └── evaluation_results.json   # Results (80% F1)
│   └── databases/
│       ├── build_dual_databases.py   # Database builder
│       ├── text_db/                  # 247MB (build locally)
│       └── code_db/                  # 93MB (build locally)
│
├── sample-smart-contract-dataset/    # 912 audit findings
├── evaluate_llm_output.py            # Ground truth tests
├── presentation_rag.tex              # LaTeX presentation
├── REFACTOR_SUMMARY.md               # Enhancement docs
└── README.md                         # This file
```

## 📊 Results

### Performance Metrics

| Metric | Value |
|--------|------:|
| **F1 Score** | **80%** |
| Precision | 80% |
| Recall | 90% |
| Tests Passed | 8/10 (80%) |

### Strengths

- ✅ **80% F1 Score** - Strong performance for RAG-based detection
- ✅ **90% Recall** - Excellent vulnerability detection capability
- ✅ **Citation-Enforced** - Every finding backed by real audit examples
- ✅ **No Data Leakage** - Test cases separate from database
- ✅ **Evidence-Based** - Transparent reasoning with [Finding N] references

### Limitations

- ⚠️ **Database Coverage** - 912 findings (can scale to 8,358 if needed)
- ⚠️ **Novel Patterns** - Can only detect similar to existing examples
- ⚠️ **Context** - Analyzes snippets in isolation (no cross-function analysis)

## 🔧 Troubleshooting

### Error: "Databases not found"

You need to build the databases locally (they're not in git):

```bash
cd new_rag_system/databases
../../venv/bin/python build_dual_databases.py
```

### Error: "Ollama connection refused"

```bash
# Start Ollama service
ollama serve

# Check models
ollama list

# Pull model if missing
ollama pull qwen2.5-coder:7b
```

### Error: "ModuleNotFoundError"

Make sure you're using the virtual environment:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📚 Documentation

- **[REFACTOR_SUMMARY.md](REFACTOR_SUMMARY.md)** - System enhancements and optimizations
- **[presentation_rag.tex](presentation_rag.tex)** - Full technical presentation (compile to PDF)
- **[new_rag_system/README.md](new_rag_system/README.md)** - System architecture details

## 🎓 Key Features

### RAG-Heavy Architecture

- **30% Structural Extraction** - Generic pattern extraction without vulnerability classification
- **40% RAG Retrieval** - Dual vector databases (GraphCodeBERT + BGE-Large)
- **30% LLM Discovery** - LLM discovers vulnerabilities from retrieved examples

### Enhancements Implemented

1. **Reduced Retrieval Values** (k=6→4) - Less noise, better focus
2. **Similarity Filtering** (60% threshold) - Quality-based retrieval
3. **Strict LLM Prompt** (ONE vuln max, 80% confidence) - Reduced false positives
4. **Missing Pattern Detection** - Detects vulnerabilities by absence (e.g., missing access control)

## 📝 License

Educational project - Smart Contract Security Analysis

---

**Project**: RAG-Heavy Vulnerability Detection System
**Date**: January 2026
