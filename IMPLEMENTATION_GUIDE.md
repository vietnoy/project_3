# Implementation Guide

Detailed guide for implementing all 3 features of the Smart Contract Security Assistant.

## Table of Contents

1. [Feature 1: Q&A System](#feature-1-qa-system)
2. [Feature 2: Code Analysis](#feature-2-code-analysis)
3. [Feature 3: Code Fixing](#feature-3-code-fixing)
4. [Adding a Web UI](#adding-a-web-ui)
5. [Advanced Features](#advanced-features)
6. [Optimization Tips](#optimization-tips)

---

## Feature 1: Q&A System

**Difficulty**: Easy ✅
**Time**: 1-2 hours

### How It Works

```
User Question → Embedding → Search Vector DB → Retrieve Top 5 Findings → LLM Answer
```

### Implementation

Already implemented in `src/chains.py`:

```python
def create_qa_chain(llm, vectorstore):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means put all context in one prompt
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Get top 5 results
        ),
        return_source_documents=True
    )
```

### Customization Options

#### 1. Change Number of Retrieved Documents

```python
# Retrieve more findings for complex questions
retriever=vectorstore.as_retriever(
    search_kwargs={"k": 10}  # Increased from 5 to 10
)
```

#### 2. Add Custom Prompt

```python
from langchain.prompts import PromptTemplate

template = """
You are a smart contract security expert assistant.

Use the following security findings to answer the question:
{context}

Question: {question}

Provide a detailed answer with:
1. Clear explanation
2. Examples if relevant
3. References to specific findings

Answer:
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_PROMPT}
)
```

#### 3. Add Conversation Memory

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Now it remembers previous questions!
qa_chain.run("What is reentrancy?")
qa_chain.run("How do I fix it?")  # Knows "it" = reentrancy
```

### Testing

```python
# In main.py or a test file
bot = SmartContractSecurityBot()

# Test questions
questions = [
    "What is a reentrancy attack?",
    "How do I validate user input?",
    "What are common Solidity security issues?",
    "Explain the difference between tx.origin and msg.sender"
]

for q in questions:
    result = bot.answer_question(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

---

## Feature 2: Code Analysis

**Difficulty**: Medium 🟡
**Time**: 4-6 hours

### How It Works

```
User Code → Search Similar Vulnerabilities → LLM Analysis → Vulnerability Report
```

### Implementation Strategy

#### Approach 1: Simple Semantic Search

```python
def analyze_code_simple(code: str, vectorstore, llm):
    # Search for similar code patterns
    similar_vulns = vectorstore.similarity_search(code, k=10)

    # Create context from findings
    context = "\n\n".join([doc.page_content for doc in similar_vulns])

    # Ask LLM to analyze
    prompt = f"""
    Analyze this Solidity code for vulnerabilities:

    {code}

    Known similar vulnerabilities:
    {context}

    List all security issues with severity levels.
    """

    analysis = llm.predict(prompt)
    return analysis
```

#### Approach 2: Multi-Step Analysis (Better)

```python
def analyze_code_advanced(code: str, vectorstore, llm):
    # Step 1: LLM identifies potential issue types
    initial_prompt = f"""
    List the types of vulnerabilities that might exist in this code:

    {code}

    Just list vulnerability types (e.g., "reentrancy", "input validation").
    """

    vuln_types = llm.predict(initial_prompt)

    # Step 2: For each type, search database
    all_findings = []
    for vuln_type in vuln_types.split('\n'):
        findings = vectorstore.similarity_search(vuln_type, k=3)
        all_findings.extend(findings)

    # Step 3: Detailed analysis with all findings
    context = "\n\n".join([doc.page_content for doc in all_findings])

    detailed_prompt = f"""
    Perform detailed security analysis on this code:

    {code}

    Reference these known vulnerabilities:
    {context}

    For each issue found, provide:
    - Line number (if identifiable)
    - Vulnerability type
    - Severity (CRITICAL/HIGH/MEDIUM/LOW)
    - Explanation
    - Similar finding ID from database
    """

    return llm.predict(detailed_prompt)
```

#### Approach 3: With Structured Output

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define output structure
response_schemas = [
    ResponseSchema(name="vulnerabilities", description="List of found vulnerabilities"),
    ResponseSchema(name="severity", description="Overall severity rating"),
    ResponseSchema(name="summary", description="Brief summary of findings")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Get format instructions
format_instructions = parser.get_format_instructions()

# Include in prompt
prompt = f"""
Analyze this code:

{code}

{format_instructions}
"""

# Parse output
result = llm.predict(prompt)
parsed = parser.parse(result)
print(parsed['vulnerabilities'])
```

### Enhanced Analysis Chain

Update `src/chains.py`:

```python
def create_analysis_chain(llm, vectorstore):
    """Enhanced code analysis with structured output"""

    def analyze_code(code: str):
        # Step 1: Identify vulnerability types
        vuln_search = llm.predict(f"List vulnerability types in: {code}")

        # Step 2: Search database
        findings = vectorstore.similarity_search(code, k=10)

        # Step 3: Detailed analysis
        analysis_prompt = f"""
        You are a smart contract security auditor.

        Code to analyze:
        ```solidity
        {code}
        ```

        Reference findings:
        {'\n'.join([f.page_content for f in findings])}

        Provide analysis in this format:

        ## Vulnerabilities Found

        ### 1. [Vulnerability Name]
        - **Severity**: CRITICAL/HIGH/MEDIUM/LOW
        - **Location**: Line X
        - **Description**: What's wrong
        - **Similar Finding**: #ID from database
        - **Impact**: What could happen

        ## Summary
        Total issues: X (Critical: X, High: X, Medium: X, Low: X)
        """

        analysis = llm.predict(analysis_prompt)

        return {
            'analysis': analysis,
            'relevant_findings': findings,
            'findings_count': len(findings)
        }

    return type('AnalysisChain', (), {'run': analyze_code})()
```

---

## Feature 3: Code Fixing

**Difficulty**: Medium-Hard 🟠
**Time**: 6-8 hours

### How It Works

```
Vulnerable Code → Analyze → Find Fix Recommendations → LLM Rewrites → Fixed Code + Explanation
```

### Implementation

#### Step 1: Retrieve Fix Recommendations

```python
def get_fix_recommendations(code: str, vectorstore):
    # Search for vulnerabilities
    vuln_findings = vectorstore.similarity_search(code, k=5)

    # Search for fix recommendations
    fix_findings = vectorstore.similarity_search(
        "fix recommendation secure code best practices",
        k=5
    )

    # Search for specific patterns
    pattern_findings = vectorstore.similarity_search(
        "checks effects interactions reentrancy guard",
        k=3
    )

    return {
        'vulnerabilities': vuln_findings,
        'fixes': fix_findings,
        'patterns': pattern_findings
    }
```

#### Step 2: Create Fix Chain

```python
def create_fix_chain(llm, vectorstore):

    fix_template = """
    You are an expert Solidity developer specializing in security.

    ## Vulnerable Code
    ```solidity
    {code}
    ```

    ## Known Vulnerabilities
    {vulnerabilities}

    ## Recommended Fixes
    {recommendations}

    ## Security Patterns
    {patterns}

    Your task:
    1. Rewrite the code to fix all security issues
    2. Apply security best practices
    3. Add comments explaining critical changes
    4. Ensure code compiles and is gas-efficient

    Provide:

    ## Fixed Code
    ```solidity
    // Fixed version with security improvements
    ```

    ## Changes Made
    1. **Change description** - Why this fix was needed (Reference: Finding #ID)
    2. **Another change** - Explanation

    ## Security Patterns Applied
    - Pattern 1
    - Pattern 2

    ## Testing Recommendations
    - What to test to verify fixes
    """

    def fix_code(code: str):
        # Get all relevant findings
        recommendations = get_fix_recommendations(code, vectorstore)

        # Format context
        vuln_context = "\n".join([
            f"- {doc.page_content[:200]}..."
            for doc in recommendations['vulnerabilities']
        ])

        fix_context = "\n".join([
            f"- {doc.page_content[:200]}..."
            for doc in recommendations['fixes']
        ])

        pattern_context = "\n".join([
            f"- {doc.page_content[:200]}..."
            for doc in recommendations['patterns']
        ])

        # Generate fix
        prompt = fix_template.format(
            code=code,
            vulnerabilities=vuln_context,
            recommendations=fix_context,
            patterns=pattern_context
        )

        result = llm.predict(prompt)

        return {
            'fixed_code': result,
            'references': recommendations
        }

    return type('FixChain', (), {'run': fix_code})()
```

#### Step 3: Extract and Compare

```python
def extract_code_blocks(text: str):
    """Extract Solidity code blocks from markdown"""
    import re
    pattern = r'```solidity\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def compare_code(original: str, fixed: str):
    """Show side-by-side comparison"""
    print("=" * 80)
    print("ORIGINAL CODE".center(40) + " | " + "FIXED CODE".center(40))
    print("=" * 80)

    orig_lines = original.split('\n')
    fixed_lines = fixed.split('\n')

    max_lines = max(len(orig_lines), len(fixed_lines))

    for i in range(max_lines):
        orig = orig_lines[i] if i < len(orig_lines) else ""
        fix = fixed_lines[i] if i < len(fixed_lines) else ""
        print(f"{orig:40} | {fix:40}")
```

---

## Adding a Web UI

### Option 1: Streamlit (Easiest)

Create `app.py`:

```python
import streamlit as st
from src.main import SmartContractSecurityBot

# Initialize bot (cache it so it doesn't reload every time)
@st.cache_resource
def load_bot():
    return SmartContractSecurityBot()

bot = load_bot()

# UI
st.title("🔒 Smart Contract Security Assistant")

tab1, tab2, tab3 = st.tabs(["Q&A", "Analyze Code", "Fix Code"])

# Tab 1: Q&A
with tab1:
    st.header("Ask Security Questions")
    question = st.text_input("Your question:")

    if st.button("Ask", key="qa"):
        with st.spinner("Thinking..."):
            result = bot.answer_question(question)
            st.write("**Answer:**")
            st.write(result['answer'])

            with st.expander("View Sources"):
                for i, doc in enumerate(result['sources'], 1):
                    st.write(f"{i}. {doc.page_content[:200]}...")

# Tab 2: Analyze
with tab2:
    st.header("Analyze Smart Contract")
    code = st.text_area("Paste your Solidity code:", height=300)

    if st.button("Analyze", key="analyze"):
        with st.spinner("Analyzing..."):
            result = bot.analyze_code(code)
            st.markdown(result['analysis'])

# Tab 3: Fix
with tab3:
    st.header("Fix Vulnerabilities")
    vuln_code = st.text_area("Paste vulnerable code:", height=300)

    if st.button("Fix", key="fix"):
        with st.spinner("Fixing..."):
            result = bot.fix_code(vuln_code)
            st.markdown(result['fixed_code'])
```

Run with:
```bash
streamlit run app.py
```

### Option 2: Gradio

```python
import gradio as gr
from src.main import SmartContractSecurityBot

bot = SmartContractSecurityBot()

def qa_interface(question):
    result = bot.answer_question(question)
    return result['answer']

def analyze_interface(code):
    result = bot.analyze_code(code)
    return result['analysis']

def fix_interface(code):
    result = bot.fix_code(code)
    return result['fixed_code']

# Create interface
with gr.Blocks() as demo:
    gr.Markdown("# Smart Contract Security Assistant")

    with gr.Tab("Q&A"):
        question_input = gr.Textbox(label="Question")
        answer_output = gr.Textbox(label="Answer")
        qa_btn = gr.Button("Ask")
        qa_btn.click(qa_interface, inputs=question_input, outputs=answer_output)

    with gr.Tab("Analyze"):
        code_input = gr.Code(language="solidity", label="Code")
        analysis_output = gr.Markdown()
        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(analyze_interface, inputs=code_input, outputs=analysis_output)

    with gr.Tab("Fix"):
        vuln_input = gr.Code(language="solidity", label="Vulnerable Code")
        fixed_output = gr.Markdown()
        fix_btn = gr.Button("Fix")
        fix_btn.click(fix_interface, inputs=vuln_input, outputs=fixed_output)

demo.launch()
```

---

## Advanced Features

### 1. Batch Analysis

```python
def analyze_multiple_contracts(contract_paths: list):
    """Analyze multiple contracts at once"""
    results = []
    for path in contract_paths:
        with open(path) as f:
            code = f.read()
        analysis = bot.analyze_code(code)
        results.append({
            'file': path,
            'analysis': analysis
        })
    return results
```

### 2. Export Reports

```python
def export_report(analysis, format='markdown'):
    """Export analysis as PDF or Markdown"""
    if format == 'markdown':
        with open('security_report.md', 'w') as f:
            f.write(analysis)
    elif format == 'pdf':
        # Use library like reportlab or weasyprint
        pass
```

### 3. Custom Vulnerability Database

```python
def add_custom_finding(finding_dict):
    """Add your own vulnerability findings"""
    # Add to vector database
    vectorstore.add_texts(
        texts=[finding_dict['content']],
        metadatas=[finding_dict]
    )
```

---

## Optimization Tips

### 1. Faster Embeddings

Use local embeddings instead of OpenAI:

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### 2. Cheaper LLM

Use GPT-3.5-turbo instead of GPT-4:

```python
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### 3. Cache Results

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_qa(question: str):
    return bot.answer_question(question)
```

### 4. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def analyze_in_parallel(codes: list):
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(bot.analyze_code, codes)
    return list(results)
```

---

## Next Steps

1. Implement Feature 1 (Q&A)
2. Test thoroughly with different questions
3. Add Feature 2 (Analysis)
4. Add Feature 3 (Fix)
5. Build web UI
6. Add advanced features

Good luck! 🚀
