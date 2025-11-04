"""
LangChain chains for different features

This module contains:
1. Q&A chain (Feature 1)
2. Code analysis chain (Feature 2)
3. Code fixing chain (Feature 3)
"""

from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate


def create_qa_chain(llm, vectorstore):
    """
    Create Q&A chain for answering security questions

    Feature 1: Simple question-answering using RAG
    """
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 similar findings
        ),
        return_source_documents=True,
        verbose=True
    )


def create_analysis_chain(llm, vectorstore):
    """
    Create code analysis chain for identifying vulnerabilities

    Feature 2: Analyze smart contract code
    """

    analysis_template = """
    You are a smart contract security expert. Analyze the following Solidity code for vulnerabilities.

    Use these known vulnerability findings as reference:
    {context}

    Smart Contract Code to Analyze:
    {code}

    Provide a detailed security analysis including:
    1. List of vulnerabilities found (with line numbers if possible)
    2. Severity level (CRITICAL/HIGH/MEDIUM/LOW)
    3. Similar findings from the database
    4. Explanation of each issue
    5. Potential impact

    Format your response clearly with sections for each vulnerability.
    """

    prompt = PromptTemplate(
        template=analysis_template,
        input_variables=["context", "code"]
    )

    # Create custom chain that retrieves relevant findings
    def analyze_code(code: str):
        # Search for similar code patterns/vulnerabilities
        relevant_docs = vectorstore.similarity_search(code, k=10)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Run LLM analysis
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(context=context, code=code)

        return {
            'analysis': result,
            'relevant_findings': relevant_docs
        }

    return type('AnalysisChain', (), {'run': analyze_code})()


def create_fix_chain(llm, vectorstore):
    """
    Create code fixing chain for rewriting vulnerable code

    Feature 3: Fix vulnerabilities in smart contract code
    """

    fix_template = """
    You are a smart contract security expert. Fix the vulnerabilities in the following Solidity code.

    Use these vulnerability findings and their recommendations as guidance:
    {context}

    Vulnerable Code:
    {code}

    Provide:
    1. **Fixed Code**: The complete corrected smart contract
    2. **Changes Made**: Detailed explanation of each fix
    3. **References**: Which findings/recommendations inspired each fix
    4. **Security Patterns Applied**: Best practices used (e.g., Checks-Effects-Interactions, ReentrancyGuard)

    Make sure the fixed code follows Solidity best practices and is production-ready.
    """

    prompt = PromptTemplate(
        template=fix_template,
        input_variables=["context", "code"]
    )

    def fix_code(code: str):
        # First, search for vulnerabilities and fixes
        vuln_docs = vectorstore.similarity_search(code, k=5)

        # Also search for fix recommendations
        fix_docs = vectorstore.similarity_search(
            "fix recommendation secure code best practices",
            k=5
        )

        # Combine all relevant context
        all_docs = vuln_docs + fix_docs
        context = "\n\n".join([doc.page_content for doc in all_docs])

        # Run LLM to fix code
        chain = LLMChain(llm=llm, prompt=prompt)
        result = chain.run(context=context, code=code)

        return {
            'fixed_code': result,
            'references': all_docs
        }

    return type('FixChain', (), {'run': fix_code})()


if __name__ == "__main__":
    # Test chains (requires database and LLM setup)
    print("This module should be imported, not run directly.")
    print("See main.py for usage examples.")
