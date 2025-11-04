# Getting Started

This guide will help you set up and run the Smart Contract Security Assistant on your Mac.

## Prerequisites

- **Python 3.10 or higher**: Check with `python3 --version`
- **Git**: Check with `git --version`
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

## Step 1: Clone the Repository

```bash
cd ~/Desktop  # or wherever you want
git clone <your-repo-url>
cd project_3
```

## Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

## Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

This will install:
- LangChain (RAG framework)
- OpenAI (LLM and embeddings)
- ChromaDB (vector database)
- Streamlit (web UI)
- And other dependencies

**Note**: This might take 5-10 minutes depending on your internet speed.

## Step 4: Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit the .env file
nano .env  # or use any text editor
```

Add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Get an API key:**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Copy and paste into `.env`

## Step 5: Verify Your Setup

```bash
# Check Python version
python --version  # Should be 3.10+

# Check installed packages
pip list | grep langchain
pip list | grep openai
pip list | grep chromadb

# Verify data exists
ls sample-smart-contract-dataset/ | head
```

You should see JSON files like `finding_62000.json`, etc.

## Step 6: Initialize the Database (First Time)

The first time you run the app, it will create embeddings for all 1000+ vulnerability findings. This is a **one-time process**.

```bash
# Test database creation
python src/database.py
```

**Expected output:**
```
Creating new database...
Loading documents...
Loaded 1000 documents
Splitting documents...
Created 2500 chunks
Creating embeddings and storing in ChromaDB...
Database created and saved to ./chroma_db
```

**Time**: 5-15 minutes depending on your internet speed
**Cost**: ~$0.01 (using OpenAI embeddings)

**Troubleshooting:**
- If you get API key error: Check your `.env` file
- If you get module import error: Make sure you activated venv
- If you get ChromaDB error: `pip install --upgrade chromadb`

## Step 7: Run the Application

### Option A: Command-Line Interface

```bash
python src/main.py
```

You'll see:
```
Smart Contract Security Assistant
==================================

Modes:
1. Ask a question
2. Analyze code
3. Fix code
Type 'quit' to exit

Select mode (1/2/3) or 'quit':
```

**Try it:**
```
Select mode: 1
Your question: What is a reentrancy attack?

Answer: [Bot explains reentrancy attacks based on your database]
```

### Option B: Streamlit Web Interface (Coming Soon)

```bash
# We'll add this in Phase 4
streamlit run app.py
```

### Option C: Jupyter Notebook (For Exploration)

```bash
jupyter notebook notebooks/exploration.ipynb
```

## Step 8: Test Each Feature

### Feature 1: Q&A

```bash
python src/main.py
# Select mode: 1
# Ask: "What are common input validation issues?"
```

### Feature 2: Code Analysis

Create a test file `test_contract.sol`:
```solidity
contract Test {
    function withdraw(uint amount) public {
        msg.sender.call{value: amount}("");
        balances[msg.sender] -= amount;
    }
}
```

```bash
python src/main.py
# Select mode: 2
# Paste the code above
# Press Ctrl+D (Mac) or Ctrl+Z (Windows)
```

### Feature 3: Fix Code

```bash
python src/main.py
# Select mode: 3
# Paste vulnerable code
# Get fixed version with explanations
```

## Common Issues & Solutions

### Issue: "No module named langchain"
**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"
**Solution:**
```bash
# Check .env file exists
cat .env

# Make sure it has your key
OPENAI_API_KEY=sk-...

# Try setting it manually
export OPENAI_API_KEY="sk-your-key"
```

### Issue: "ChromaDB persistence error"
**Solution:**
```bash
# Delete and recreate
rm -rf chroma_db/
python src/database.py
```

### Issue: "Rate limit exceeded"
**Solution:**
- You're hitting OpenAI API limits
- Wait a few minutes or upgrade your OpenAI plan
- Or use free embeddings: `pip install sentence-transformers`

## Project Structure Overview

```
project_3/
├── src/
│   ├── main.py          # Run this to start the bot
│   ├── database.py      # Vector database setup
│   ├── chains.py        # LangChain chains (Q&A, Analysis, Fix)
│   └── utils.py         # Helper functions
│
├── sample-smart-contract-dataset/  # Your 1000+ vulnerability findings
├── chroma_db/           # Vector database (created on first run)
├── .env                 # Your API keys (don't commit!)
└── requirements.txt     # Python dependencies
```

## Next Steps

1. **Explore the code**: Read through `src/main.py` to understand the flow
2. **Try all 3 features**: Q&A, analysis, fixing
3. **Read IMPLEMENTATION_GUIDE.md**: Learn how to extend features
4. **Customize prompts**: Edit `src/chains.py` to improve responses
5. **Build a UI**: Follow Phase 4 to add Streamlit interface

## Cost Monitoring

**OpenAI Usage:**
- Embeddings (one-time): ~$0.01
- Each Q&A: ~$0.01
- Each analysis: ~$0.02
- Each fix: ~$0.03-0.05

**For 100 queries: ~$1-3**

**Check your usage:**
- [platform.openai.com/usage](https://platform.openai.com/usage)

## Free Alternatives

If you want to avoid costs:

1. **Free Embeddings:**
```bash
# In src/database.py, replace:
from langchain.embeddings import OpenAIEmbeddings
# with:
from langchain.embeddings import HuggingFaceEmbeddings
```

2. **Free LLM (needs GPU):**
```bash
# Install transformers
pip install transformers torch

# In src/main.py, replace:
from langchain.chat_models import ChatOpenAI
# with:
from langchain.llms import HuggingFacePipeline
```

## Getting Help

- **LangChain Docs**: [python.langchain.com](https://python.langchain.com/)
- **OpenAI API Docs**: [platform.openai.com/docs](https://platform.openai.com/docs/)
- **ChromaDB Docs**: [docs.trychroma.com](https://docs.trychroma.com/)

## Ready to Build!

You're all set! Start with Feature 1 (Q&A) and gradually add more capabilities.

**Quick start:**
```bash
source venv/bin/activate
python src/main.py
```

Good luck! 🚀
