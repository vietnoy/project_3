"""
Main application for Smart Contract Security Assistant

This is the entry point for the RAG-based chatbot that provides:
1. Q&A about smart contract security
2. Code analysis for vulnerabilities
3. Automated code fixing
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from database import load_vulnerability_database
from chains import create_qa_chain, create_analysis_chain, create_fix_chain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SmartContractSecurityBot:
    """Main chatbot class with 3 core features"""

    def __init__(self):
        """Initialize the bot with vector database and LLM"""
        print("Loading vulnerability database...")
        self.vectorstore = load_vulnerability_database()

        print("Initializing LLM...")
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        print("Creating chains...")
        self.qa_chain = create_qa_chain(self.llm, self.vectorstore)
        self.analysis_chain = create_analysis_chain(self.llm, self.vectorstore)
        self.fix_chain = create_fix_chain(self.llm, self.vectorstore)

        print("Bot ready!")

    def answer_question(self, question: str) -> dict:
        """
        Feature 1: Answer questions about smart contract security

        Args:
            question: User's security question

        Returns:
            dict with 'answer' and 'sources'
        """
        result = self.qa_chain(question)
        return {
            'answer': result['result'],
            'sources': result.get('source_documents', [])
        }

    def analyze_code(self, code: str) -> dict:
        """
        Feature 2: Analyze smart contract code for vulnerabilities

        Args:
            code: Solidity smart contract code

        Returns:
            dict with vulnerability analysis
        """
        return self.analysis_chain.run(code)

    def fix_code(self, code: str) -> dict:
        """
        Feature 3: Fix vulnerabilities in smart contract code

        Args:
            code: Vulnerable Solidity code

        Returns:
            dict with fixed code and explanations
        """
        return self.fix_chain.run(code)


def main():
    """Main function - simple CLI interface"""
    bot = SmartContractSecurityBot()

    print("\n" + "="*60)
    print("Smart Contract Security Assistant")
    print("="*60)
    print("\nModes:")
    print("1. Ask a question")
    print("2. Analyze code")
    print("3. Fix code")
    print("Type 'quit' to exit\n")

    while True:
        mode = input("Select mode (1/2/3) or 'quit': ").strip()

        if mode.lower() == 'quit':
            print("Goodbye!")
            break

        if mode == '1':
            question = input("\nYour question: ")
            result = bot.answer_question(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {len(result['sources'])} findings used")

        elif mode == '2':
            print("\nPaste your code (press Ctrl+D or Ctrl+Z when done):")
            code_lines = []
            try:
                while True:
                    line = input()
                    code_lines.append(line)
            except EOFError:
                pass
            code = '\n'.join(code_lines)

            result = bot.analyze_code(code)
            print(f"\nAnalysis:\n{result}")

        elif mode == '3':
            print("\nPaste your vulnerable code (press Ctrl+D or Ctrl+Z when done):")
            code_lines = []
            try:
                while True:
                    line = input()
                    code_lines.append(line)
            except EOFError:
                pass
            code = '\n'.join(code_lines)

            result = bot.fix_code(code)
            print(f"\nFixed Code:\n{result}")

        else:
            print("Invalid mode. Please select 1, 2, or 3.")

        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    main()
