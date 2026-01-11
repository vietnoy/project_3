# Quick Start Guide - RAG-Heavy Vulnerability Detector

## What You Have

A TRUE RAG-heavy vulnerability detection system where:
- **30%** intelligence from structural pattern extraction
- **70%** intelligence from RAG (retrieval + LLM discovery)

## Files Created

```
new_rag_system/
├── databases/
│   ├── build_dual_databases.py    # Build GraphCodeBERT + BGE databases
│   └── README.md
├── evaluation/
│   └── run_evaluation.py          # Proper LLM output evaluation
├── structural_extractor.py         # Minimal feature extraction (NO vuln names!)
├── rag_retriever.py               # Heavy retrieval from dual databases
├── rag_detector.py                # Main detector with discovery-focused LLM
├── QUICK_START.md                 # This file
└── README.md                      # Full documentation
```

## Step-by-Step Setup

### Step 1: Build Databases (10-20 minutes)

```bash
cd new_rag_system/databases/
python build_dual_databases.py
```

This will create:
- `text_db/` - BGE-Large embeddings for pattern matching
- `code_db/` - GraphCodeBERT embeddings for code similarity

From 912 curated audit findings.

### Step 2: Test the Detector

```bash
cd new_rag_system/
python rag_detector.py
```

This runs a simple test to verify everything works.

### Step 3: Run Full Evaluation (30-60 minutes)

```bash
cd new_rag_system/evaluation/
python run_evaluation.py
```

This tests the system with 10 ground truth test cases and measures:
- **Precision**: % of reported vulnerabilities that are real
- **Recall**: % of real vulnerabilities detected
- **F1 Score**: Overall accuracy

**Note**: LLM inference is slow - budget 30-60 minutes for full evaluation.

### Step 4: Analyze Results

Check `evaluation/evaluation_results.json` for metrics.

**Expected Performance:**
- F1 Score: 70-85% (good to excellent)
- Precision: 75-90%
- Recall: 70-85%

If lower:
- Increase k_code and k_text in run_evaluation.py
- Check if databases built correctly
- Verify Ollama model is running

## How It Works

### Example Flow:

**Input Code:**
```solidity
function withdraw() public {
    uint256 amount = balances[msg.sender];
    msg.sender.call{value: amount}("");
    balances[msg.sender] = 0;
}
```

**Step 1: Structural Extraction**
```
Extracted:
- External call at line 3 (type: call, has_value: True)
- State change at line 4 (variable: balances, operation: assignment)
- Ordering: call_before_state_change = TRUE
- Function: withdraw, visibility: public, modifiers: []
```

**Step 2: RAG Retrieval**
```
Retrieved 12 findings:
- 6 from code similarity (similar structure)
- 6 from pattern matching ("external call before state update")
```

**Step 3: LLM Discovery**
```
LLM compares user code to retrieved findings and discovers:
"This code matches reentrancy pattern [Finding 1], [Finding 3]
because external call occurs before state update, allowing
callback attack..."
```

**Key**: LLM discovers "reentrancy" by comparing to examples,
NOT because we told it to look for reentrancy!

## Key Differences from Old System

| Aspect | Old System | New RAG System |
|--------|------------|---------------|
| Feature Extraction | Returns "reentrancy", "access-control" | Returns "external_call", "state_change" |
| Intelligence | 70% features, 30% RAG | 30% features, 70% RAG |
| LLM Role | Cite pre-detected vulns | Discover vulns from examples |
| Evaluation | Tests feature extraction | Tests LLM output quality |
| Code Quality | Emojis, complex | Clean, minimal |

## Tuning Parameters

In `evaluation/run_evaluation.py`:

```python
k_code = 6  # Number of code similarities (increase for more context)
k_text = 6  # Number of pattern matches (increase for more context)
```

Higher k = More findings retrieved = Better discovery (but slower)

Recommended ranges:
- Fast testing: k_code=3, k_text=3
- Normal: k_code=6, k_text=6 (default)
- Thorough: k_code=10, k_text=10

## Troubleshooting

### "Database not found"
```bash
cd databases/
python build_dual_databases.py
```

### "Ollama model not found"
```bash
ollama pull qwen2.5-coder:7b
```

### "Low F1 score"
1. Check databases built correctly (should be ~80-130 MB total)
2. Increase k values
3. Verify 912 findings in sample-smart-contract-dataset/

### "Evaluation too slow"
- Use smaller model: `qwen2.5-coder:1.5b` or `gemma:2b`
- Reduce number of test cases in evaluate_llm_output.py
- Run on fewer test cases first to verify

## Next Steps

1. Build databases
2. Run evaluation
3. Analyze results
4. If good (F1 > 70%), expand dataset to 8,000+ findings
5. Re-build databases with larger dataset
6. Compare performance

## Philosophy

This system follows TRUE RAG principles:
- Minimal feature engineering
- Heavy reliance on data/retrieval
- LLM discovers patterns from examples
- Proper evaluation of end-to-end quality

Start with quality (912 findings), get it working well,
then scale up to quantity (8,000+ findings).

## Support

- Check README.md for detailed documentation
- Check databases/README.md for database details
- Original evaluation framework: ../../evaluate_llm_output.py
