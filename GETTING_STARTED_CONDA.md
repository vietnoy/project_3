# Getting Started (Conda Version)

This guide uses **Conda** instead of venv for environment management.

## Prerequisites

- **Python 3.10 or higher**: Check with `python --version`
- **Conda/Anaconda/Miniconda**: Check with `conda --version`
- **Git**: Check with `git --version`
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

## Step 1: Clone the Repository

### On Mac (after pushing from Windows):
```bash
cd ~/Desktop  # or wherever you want
git clone https://github.com/YOUR-USERNAME/smart-contract-security-rag.git
cd smart-contract-security-rag
```

### On Windows (if testing locally):
```bash
cd "c:\Users\admin\Desktop\project_3"
```

## Step 2: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n smart-contract-rag python=3.10 -y

# Activate the environment
conda activate smart-contract-rag

# You should see (smart-contract-rag) in your terminal prompt
```

**Note**: You'll need to activate this environment every time you work on the project.

## Step 3: Install Dependencies

```bash
# Make sure environment is activated
conda activate smart-contract-rag

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

**Time**: 5-10 minutes depending on internet speed

## Step 4: Set Up Environment Variables

```bash
# Copy the example env file
cp .env.example .env

# Edit the .env file
# Mac/Linux:
nano .env
# or use any text editor

# Windows:
notepad .env
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
# Make sure conda environment is activated
conda activate smart-contract-rag

# Check Python version
python --version  # Should be 3.10+

# Check installed packages
conda list | grep langchain
conda list | grep openai
conda list | grep chromadb

# Verify data exists
ls sample-smart-contract-dataset/ | head
```

## Step 6: Initialize the Database (First Time)

The first time you run the app, it will create embeddings for all 1000+ vulnerability findings. This is a **one-time process**.

```bash
# Make sure environment is activated
conda activate smart-contract-rag

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

**Time**: 5-15 minutes depending on internet speed
**Cost**: ~$0.01 (using OpenAI embeddings)

**Troubleshooting:**
- If you get API key error: Check your `.env` file
- If you get module import error: Make sure conda environment is activated
- If you get ChromaDB error: `pip install --upgrade chromadb`

## Step 7: Run the Application

```bash
# Make sure environment is activated
conda activate smart-contract-rag

# Run the CLI application
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
# Press Ctrl+D (Mac/Linux) or Ctrl+Z then Enter (Windows)
```

### Feature 3: Fix Code

```bash
python src/main.py
# Select mode: 3
# Paste vulnerable code
# Get fixed version with explanations
```

## Common Issues & Solutions

### Issue: "conda: command not found"
**Solution:**
```bash
# Install Miniconda (lightweight) or Anaconda
# Mac:
brew install miniconda
# or download from: https://docs.conda.io/en/latest/miniconda.html

# Windows:
# Download installer from: https://docs.conda.io/en/latest/miniconda.html
```

### Issue: "No module named langchain"
**Solution:**
```bash
# Make sure environment is activated
conda activate smart-contract-rag

# Reinstall
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"
**Solution:**
```bash
# Check .env file exists
cat .env  # Mac/Linux
type .env  # Windows

# Make sure it has your key
OPENAI_API_KEY=sk-...

# Or set it in environment
export OPENAI_API_KEY="sk-your-key"  # Mac/Linux
set OPENAI_API_KEY=sk-your-key  # Windows
```

### Issue: Environment not activating
**Solution:**
```bash
# Initialize conda for your shell
conda init bash  # Mac/Linux
conda init powershell  # Windows PowerShell
conda init cmd.exe  # Windows Command Prompt

# Restart your terminal, then:
conda activate smart-contract-rag
```

### Issue: "ChromaDB persistence error"
**Solution:**
```bash
# Delete and recreate
rm -rf chroma_db/
python src/database.py
```

## Conda Environment Management

### Daily Usage
```bash
# Every time you start working:
conda activate smart-contract-rag

# Run your code
python src/main.py

# When done:
conda deactivate
```

### View Environments
```bash
# List all conda environments
conda env list

# See installed packages
conda list
```

### Update Environment
```bash
# Update a specific package
conda activate smart-contract-rag
pip install --upgrade langchain

# Update all packages
pip install --upgrade -r requirements.txt
```

### Export Environment
```bash
# Create environment.yml for sharing
conda env export > environment.yml

# Others can recreate with:
conda env create -f environment.yml
```

### Remove Environment
```bash
# If you need to start over
conda deactivate
conda env remove -n smart-contract-rag

# Then recreate
conda create -n smart-contract-rag python=3.10 -y
```

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

## Using Jupyter Notebooks

```bash
# Make sure environment is activated
conda activate smart-contract-rag

# Start Jupyter
jupyter notebook

# Open notebooks/exploration.ipynb
# or create a new notebook
```

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
- **Conda Docs**: [docs.conda.io](https://docs.conda.io/)

## Quick Start Command Summary

```bash
# Setup (one time)
conda create -n smart-contract-rag python=3.10 -y
conda activate smart-contract-rag
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key

# Initialize database (one time)
python src/database.py

# Run the app (every time)
conda activate smart-contract-rag
python src/main.py
```

## Ready to Build!

You're all set! Start with Feature 1 (Q&A) and gradually add more capabilities.

Good luck! 🚀
