# Smart Contract Security Assistant with RAG

A chatbot powered by Retrieval-Augmented Generation (RAG) that helps analyze smart contract vulnerabilities, answer security questions, and fix vulnerable code.

## Project Overview

This project uses a dataset of 1000+ smart contract security audit findings to build an AI assistant with three core features:

1. **Q&A**: Answer questions about smart contract security
2. **Code Analysis**: Identify vulnerabilities in user-submitted code
3. **Code Fixing**: Automatically fix vulnerable code with explanations

## Why RAG (Not Finetuning)?

We chose RAG over finetuning because:

- ✅ **Cost-effective**: Much cheaper than finetuning
- ✅ **Your data is knowledge-based**: Perfect for retrieval
- ✅ **LLMs already code**: They just need domain-specific security knowledge
- ✅ **Flexible & updatable**: Add new vulnerabilities without retraining
- ✅ **Better transparency**: See which findings the LLM uses
- ✅ **Works with smaller datasets**: ~1000 findings is perfect for RAG

**When to use finetuning instead:**
- You need to change model behavior/style
- You have 10,000+ training examples
- You need the model to memorize specific patterns
- You have budget for GPU training costs

## How RAG Works

```
User Question → Retrieve Relevant Docs → LLM Generates Answer
               (from vector database)   (using retrieved context)
```

**Example flow:**
```
User: "How do I prevent reentrancy attacks?"
  ↓
1. Convert question to vector embedding
2. Search 1000+ vulnerability findings
3. Find top 5 most similar findings about reentrancy
4. Give those findings + question to LLM
5. LLM generates answer based on actual audit data
```

## What is LangChain?

LangChain is a framework that makes it easier to build LLM applications. Think of it as React for AI apps.

**Without LangChain**: 200+ lines of manual code
**With LangChain**: ~30 lines using pre-built components

**LangChain provides:**
- Document loaders (JSON, PDF, CSV, etc.)
- Text splitters (chunk large documents)
- Embeddings (OpenAI, HuggingFace, etc.)
- Vector stores (ChromaDB, FAISS, Pinecone)
- LLM integrations (OpenAI, Claude, Llama)
- Chains (pre-built workflows)
- Memory (conversation history)
- Agents (LLMs that use tools)

## Architecture

```
┌─────────────────────────────────────────────────┐
│           USER INTERFACE                        │
│  [Chat Input] [Code Input Box] [Submit]        │
└────────────────┬────────────────────────────────┘
                 │
         User selects mode:
         1. Ask Question
         2. Analyze Code
         3. Fix Code
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           FEATURE ROUTER                        │
│  - Q&A → RetrievalQA Chain                     │
│  - Analysis → Code Analysis Chain               │
│  - Fix → Code Fixing Chain                      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         VECTOR DATABASE (ChromaDB)              │
│  1000+ Smart Contract Vulnerability Findings    │
│  - Embeddings of titles, content, fixes        │
│  - Metadata: severity, type, recommendations   │
└────────────────┬────────────────────────────────┘
                 │
         Retrieves relevant findings
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│              LLM (GPT-4 / Claude)               │
│  - Answers questions using findings             │
│  - Analyzes code against known vulnerabilities  │
│  - Generates fixed code with explanations       │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│           RESPONSE TO USER                      │
│  - Natural language answer (Q&A)                │
│  - Vulnerability report (Analysis)              │
│  - Fixed code + explanation (Fix)               │
└─────────────────────────────────────────────────┘
```

## Tech Stack

### Core Technologies
- **Python 3.10+**: Programming language
- **LangChain**: RAG framework
- **OpenAI API**: LLM (GPT-4) and embeddings
- **ChromaDB**: Vector database (local, free)

### UI Options (choose one)
- **Streamlit**: Easiest - web app in pure Python
- **Gradio**: Great for ML demos
- **Chainlit**: Built specifically for chatbots
- **Flask/FastAPI**: Full control, more complex

### Optional
- **sentence-transformers**: Free open-source embeddings
- **Llama 2/3 or Mistral**: Free open-source LLMs (needs GPU)

## Features in Detail

### Feature 1: Q&A (Easy ✅)

**What it does**: Answer questions about smart contract security

**Example questions:**
- "What is a reentrancy attack?"
- "How do I validate user input in smart contracts?"
- "What are common Solidity security issues?"
- "Explain the difference between tx.origin and msg.sender"

**Implementation:**
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

result = qa_chain("What is a reentrancy attack?")
```

### Feature 2: Code Analysis (Medium 🟡)

**What it does**: Identify vulnerabilities in user-submitted smart contract code

**Example:**
```
User pastes:
  contract MyToken {
      function withdraw(uint amount) public {
          msg.sender.call{value: amount}("");
          balances[msg.sender] -= amount;
      }
  }

Bot responds:
  ⚠️ CRITICAL: Reentrancy Vulnerability Detected

  Line 3: msg.sender.call{value: amount}("");
  Issue: State update happens AFTER external call
  Similar to: Finding #62123 "Reentrancy in withdrawal function"

  Severity: CRITICAL
```

### Feature 3: Fixing Vulnerabilities (Medium-Hard 🟠)

**What it does**: Rewrite vulnerable code to fix security issues

**Example:**
```
Input: Vulnerable code
Output:
  1. Fixed code with proper security patterns
  2. Explanation of each change
  3. References to findings that inspired fixes
```

## Dataset Structure

Your `sample-smart-contract-dataset/` contains 1000+ JSON files like:

```json
{
  "id": 62000,
  "title": "Missing Input Validation in REST Api Wrapper Function",
  "content": "The function does not validate input...",
  "summary": "Brief summary of the vulnerability",
  "impact": "MEDIUM",
  "recommendation": "Validate all input parameters..."
}
```

## Cost Estimates

### Using OpenAI:

**One-time setup (embeddings):**
- 1000 findings × 500 tokens = 500,000 tokens
- text-embedding-3-small: $0.02 per 1M tokens
- **Cost: ~$0.01** (basically free)

**Per-query (GPT-4):**
- Feature 1 (Q&A): ~$0.01 per question
- Feature 2 (Analysis): ~$0.02 per analysis
- Feature 3 (Fix): ~$0.03-0.05 per fix

**100 users × 10 queries each: ~$10-30 total**

### Free/Cheap Alternatives:
- **Embeddings**: sentence-transformers (free, local)
- **LLM**: GPT-3.5-turbo ($0.001/query) or Llama 2/3 (free, needs GPU)
- **Vector DB**: ChromaDB local (free)

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- ✅ Set up Python environment
- ✅ Install dependencies (LangChain, OpenAI, ChromaDB)
- ✅ Load JSON files into vector database
- ✅ Test basic embedding & retrieval
- ✅ Build simple Q&A (Feature 1)

### Phase 2: Code Analysis (Week 2)
- ✅ Accept code input from user
- ✅ Implement semantic search for code patterns
- ✅ Generate vulnerability reports
- ✅ Link findings to database entries

### Phase 3: Code Fixing (Week 3)
- ✅ Retrieve fix recommendations from database
- ✅ Implement code rewriting with LLM
- ✅ Add before/after comparison
- ✅ Explain each change

### Phase 4: Polish (Week 4)
- ✅ Build web interface (Streamlit/Gradio)
- ✅ Add code syntax highlighting
- ✅ Show source findings used
- ✅ Add severity ratings

## Getting Started

See [GETTING_STARTED.md](./GETTING_STARTED.md) for detailed setup instructions.

## Project Structure

```
project_3/
├── README.md                           # This file
├── GETTING_STARTED.md                  # Setup instructions
├── IMPLEMENTATION_GUIDE.md             # Detailed implementation guide
├── requirements.txt                    # Python dependencies
├── .env.example                        # Example environment variables
├── .gitignore                          # Git ignore file
├── sample-smart-contract-dataset/      # Your vulnerability findings (1000+ JSON files)
├── src/
│   ├── __init__.py
│   ├── main.py                         # Main application
│   ├── database.py                     # Vector database setup
│   ├── chains.py                       # LangChain chains (Q&A, Analysis, Fix)
│   └── utils.py                        # Utility functions
├── notebooks/
│   └── exploration.ipynb               # Jupyter notebook for testing
└── tests/
    └── test_basic.py                   # Unit tests
```

## Key Concepts Summary

### RAG (Retrieval-Augmented Generation)
- Combines retrieval (searching documents) with generation (LLM)
- Like an open-book exam for AI
- Perfect for knowledge-based applications

### Embeddings
- Convert text to numerical vectors
- Similar meanings = similar vectors
- Enable semantic search

### Vector Database
- Stores embeddings
- Fast similarity search
- Returns most relevant documents

### LangChain
- Framework for building LLM apps
- Pre-built components save time
- Easy to swap LLMs and vector DBs

## Next Steps

1. **Clone this repo on your Mac**
2. **Follow GETTING_STARTED.md** to set up environment
3. **Run the basic Q&A feature** first
4. **Expand to code analysis** and fixing

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [RAG Explained](https://www.pinecone.io/learn/retrieval-augmented-generation/)

## Questions Covered in Our Discussion

✅ What is RAG vs Finetuning?
✅ What is LangChain for?
✅ How do embeddings work?
✅ What is a vector database?
✅ How to implement Q&A, code analysis, and fixing?
✅ What technologies to use?
✅ Cost estimates
✅ Architecture design

## Contact

Built as a learning project for understanding RAG and LLM applications in smart contract security.

---

**Ready to build!** Start with `GETTING_STARTED.md` on your Mac.
