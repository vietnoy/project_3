# Demo Guide - Viewing Test Cases and LLM Analysis

This guide shows you how to export test cases and run demos to see LLM analysis outputs.

## 📦 Quick Start

### 1. Export Test Cases to JSON

Export all 10 test cases to readable JSON files:

```bash
cd new_rag_system/evaluation
../../venv/bin/python export_test_cases.py
```

**Output:**
- Creates `test_cases_export/` folder
- 10 JSON files: `test_01_*.json`, `test_02_*.json`, etc.
- 1 index file: `00_index.json`

**Example test case file (`test_01_reentrancy_in_withdraw_function.json`):**
```json
{
  "test_number": 1,
  "name": "Reentrancy in withdraw function",
  "code": "function withdraw() public {\n  uint256 amount = balances[msg.sender];\n  msg.sender.call{value: amount}(\"\");\n  balances[msg.sender] = 0;\n}",
  "should_have_zero_vulns": false,
  "expected_vulnerabilities": [
    {
      "type": "reentrancy",
      "severity": "HIGH",
      "must_contain_keywords": ["external call", "state", "reentrancy"],
      "description": "External call before state update allows reentrant calls"
    }
  ]
}
```

### 2. Run Demo Detection with LLM Output Saving

Run detection on test cases and save full LLM analysis:

```bash
cd new_rag_system/evaluation

# Run ALL test cases (takes ~5 minutes)
../../venv/bin/python demo_detection.py

# Or run specific test cases only (faster)
../../venv/bin/python demo_detection.py --tests 1 3 5
```

**Output:**
- Creates `demo_outputs/` folder
- For each test: 2 files
  - `demo_XX_*.json` - Full detection result with metadata
  - `demo_XX_*_llm_output.txt` - LLM analysis in plain text
- 1 index file: `00_demo_index.json`

**Example LLM output file (`demo_01_reentrancy_in_withdraw_function_llm_output.txt`):**
```
TEST CASE: Reentrancy in withdraw function
================================================================================

CODE:
--------------------------------------------------------------------------------
function withdraw() public {
  uint256 amount = balances[msg.sender];
  msg.sender.call{value: amount}("");
  balances[msg.sender] = 0;
}
--------------------------------------------------------------------------------

LLM ANALYSIS OUTPUT:
================================================================================

### Reentrancy Attack - HIGH

**Finding Reference:** [Finding solodit_7425] (87% similarity)

**Structural Match:**
Your code structure matches the vulnerable pattern found in [Finding solodit_7425]:
- Line 2: External call (msg.sender.call) transfers control to untrusted code
- Line 3: State update (balances[msg.sender] = 0) happens AFTER the call
- This violates Checks-Effects-Interactions (CEI) pattern

**Why This Is Vulnerable:**
The external call can trigger a callback to the caller's contract.
During this callback, balances[msg.sender] still contains the old value,
allowing the attacker to re-enter withdraw() and drain funds repeatedly.

**Recommendation:**
Move state update before external call:
  balances[msg.sender] = 0;  // State change FIRST
  msg.sender.call{value: amount}("");  // External call LAST

**Confidence:** 95% (high structural similarity with known reentrancy pattern)
```

## 🎯 Use Cases

### For Presentations
1. Show test cases in JSON format (matches database structure)
2. Display LLM analysis output as proof of detection
3. Demonstrate citation-based reasoning

### For Debugging
1. Check what patterns were extracted
2. See which findings were retrieved
3. Understand why detection succeeded/failed

### For Documentation
1. Include test cases in reports
2. Show concrete examples of LLM reasoning
3. Validate evaluation methodology

## 📋 Available Options

### Export Test Cases

```bash
# Export to custom directory
../../venv/bin/python export_test_cases.py --output ./my_tests
```

### Run Demo Detection

```bash
# Run specific tests
../../venv/bin/python demo_detection.py --tests 1 2 3

# Custom output directory
../../venv/bin/python demo_detection.py --output ./my_demos

# Quiet mode (less verbose)
../../venv/bin/python demo_detection.py --quiet

# Custom database paths
../../venv/bin/python demo_detection.py \
  --code-db /path/to/code_db \
  --text-db /path/to/text_db
```

## 📂 Output Structure

```
new_rag_system/evaluation/
├── test_cases_export/          # Test cases in JSON
│   ├── 00_index.json           # Index of all tests
│   ├── test_01_*.json          # Test case 1
│   ├── test_02_*.json          # Test case 2
│   └── ...
│
└── demo_outputs/               # Detection results
    ├── 00_demo_index.json      # Index of all demos
    ├── demo_01_*.json          # Full detection result
    ├── demo_01_*_llm_output.txt # LLM analysis only
    ├── demo_02_*.json
    ├── demo_02_*_llm_output.txt
    └── ...
```

## 🔍 What's in the Files?

### Test Case JSON (`test_XX_*.json`)
- Test number and name
- Code snippet
- Expected vulnerabilities
- Required keywords for validation
- Severity levels

### Demo JSON (`demo_XX_*.json`)
- Test info (name, code, timestamp)
- Expected vulnerabilities
- Structural patterns extracted
- Retrieved findings (with similarity scores)
- **Full LLM analysis**
- Metadata (timing, configuration)

### LLM Output TXT (`demo_XX_*_llm_output.txt`)
- Test case code
- **Complete LLM analysis in markdown format**
- Easy to read and copy-paste

## 💡 Tips

1. **Run specific tests first** to check if system is working:
   ```bash
   ../../venv/bin/python demo_detection.py --tests 3
   ```

2. **Compare outputs** between different test cases to see how LLM reasoning varies

3. **Use TXT files** for quick viewing, **JSON files** for programmatic access

4. **Include in slides** - paste LLM output directly into presentation

5. **Verify no data leakage** - test case code should NOT appear in database

## ⚠️ Requirements

- Databases must be built first: `../../venv/bin/python ../databases/build_dual_databases.py`
- Ollama must be running: `ollama serve`
- Model must be pulled: `ollama pull qwen2.5-coder:7b`

## 🚀 Quick Demo for Presentation

```bash
# 1. Export test cases (fast)
../../venv/bin/python export_test_cases.py

# 2. Run demo on one interesting test (3 = Unchecked return value)
../../venv/bin/python demo_detection.py --tests 3

# 3. Open the output files
open demo_outputs/demo_03_*_llm_output.txt
open test_cases_export/test_03_*.json

# Now you have:
# - Test case in JSON (input)
# - LLM analysis in TXT (output)
# Perfect for live demonstration!
```
