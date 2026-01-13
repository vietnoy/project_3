"""
RAG Vulnerability Detector - Discovery-Focused Analysis
=======================================================

Main detector that ties everything together:
1. Structural extraction (minimal)
2. Heavy RAG retrieval
3. LLM discovery of vulnerabilities

Philosophy: Let the LLM DISCOVER vulnerabilities by comparing
user code to similar vulnerable examples from the database.
"""
from typing import Dict, List
from langchain_community.llms import Ollama

from structural_extractor import StructuralPatternExtractor
from rag_retriever import RAGRetriever


class RAGVulnerabilityDetector:
    """
    True RAG-heavy vulnerability detector.

    Intelligence distribution:
    - Structural extraction: 30%
    - RAG retrieval: 40%
    - LLM discovery: 30%
    """

    def __init__(
        self,
        code_db_path: str = './databases/code_db',
        text_db_path: str = './databases/text_db',
        llm_model: str = 'qwen2.5-coder:7b',
        k_code: int = 4,
        k_text: int = 4,
        min_similarity: float = 0.60
    ):
        """
        Initialize RAG detector

        Args:
            code_db_path: Path to code database
            text_db_path: Path to text database
            llm_model: Ollama model name
            k_code: Number of code similarities to retrieve (default: 3)
            k_text: Number of text patterns to retrieve (default: 3)
            min_similarity: Minimum similarity threshold for filtering (default: 0.65)
        """
        print("Initializing RAG Vulnerability Detector...")
        print("=" * 80)

        # Initialize components
        self.extractor = StructuralPatternExtractor()
        self.retriever = RAGRetriever(code_db_path, text_db_path, min_similarity=min_similarity)

        print(f"Loading LLM: {llm_model}...")
        self.llm = Ollama(model=llm_model, temperature=0.2)

        self.k_code = k_code
        self.k_text = k_text
        self.min_similarity = min_similarity

        print("=" * 80)
        print("RAG Detector ready")
        print()

    def detect(self, code: str, verbose: bool = True) -> Dict:
        """
        Analyze code for vulnerabilities using RAG approach

        Args:
            code: Solidity code to analyze
            verbose: Print progress

        Returns:
            Dict with:
                - structural_patterns: Extracted patterns
                - retrieved_findings: Similar findings from database
                - analysis: LLM analysis discovering vulnerabilities
                - metadata: Detection metadata
        """
        if verbose:
            print()
            print("=" * 80)
            print("RAG VULNERABILITY DETECTION")
            print("=" * 80)
            print()

        # Step 1: Extract structural patterns (NO vulnerability classification)
        if verbose:
            print("Step 1: Structural Pattern Extraction")
            print("-" * 80)

        patterns = self.extractor.extract_patterns(code)

        if verbose:
            print(f"Extracted patterns:")
            print(f"  External calls: {len(patterns['external_calls'])}")
            print(f"  State changes: {len(patterns['state_changes'])}")
            print(f"  Functions: {len(patterns['functions'])}")
            print(f"  Control structures: {len(patterns['control_structures'])}")
            print(f"  Dangerous operations: {len(patterns['dangerous_operations'])}")

            if patterns['ordering_patterns']:
                print(f"  Ordering patterns:")
                for key, value in patterns['ordering_patterns'].items():
                    if value and isinstance(value, bool):
                        print(f"    - {key}")

            print()

        # Step 2: Retrieve similar findings (HEAVY RAG)
        if verbose:
            print("Step 2: Retrieval from Vector Databases")
            print("-" * 80)

        findings = self.retriever.retrieve(
            code=code,
            structural_patterns=patterns,
            k_code=self.k_code,
            k_text=self.k_text,
            verbose=verbose
        )

        # Step 3: LLM analyzes and DISCOVERS vulnerabilities
        if verbose:
            print("Step 3: LLM Analysis - Discovering Vulnerabilities")
            print("-" * 80)
            print("Sending to LLM for analysis...")
            print()

        analysis = self._analyze_with_llm(code, patterns, findings, verbose)

        if verbose:
            print("=" * 80)
            print("Detection Complete")
            print("=" * 80)
            print()

        return {
            'structural_patterns': patterns,
            'retrieved_findings': findings,
            'analysis': analysis,
            'metadata': {
                'num_findings_retrieved': len(findings),
                'num_external_calls': len(patterns['external_calls']),
                'num_state_changes': len(patterns['state_changes']),
                'k_code': self.k_code,
                'k_text': self.k_text
            }
        }

    def _analyze_with_llm(
        self,
        code: str,
        patterns: Dict,
        findings: List[Dict],
        verbose: bool
    ) -> str:
        """
        LLM analyzes code by comparing to similar vulnerable examples.

        CRITICAL: Prompt emphasizes DISCOVERY not CONFIRMATION.
        """
        # Build findings context
        findings_text = self._format_findings_for_llm(findings)

        # Build structural patterns text
        patterns_text = self._format_patterns_for_llm(patterns)

        # Strict discovery-focused prompt
        prompt = f"""You are a smart contract security expert performing RAG-based vulnerability analysis.

**STRICT INSTRUCTIONS - READ CAREFULLY:**

1. Report ONLY ONE vulnerability - the MOST CRITICAL match between user code and findings
2. The vulnerability MUST exist in BOTH the user's code AND the cited finding
3. ONLY report if you find a DIRECT structural match (same pattern, same order)
4. Do NOT report vulnerabilities mentioned in findings if they don't exist in user code
5. If unsure or no strong match exists, report "No matching vulnerabilities detected"

**User's Code:**
```solidity
{code}
```

**Structural Patterns Detected:**
{patterns_text}

**Similar Vulnerable Code from Database ({len(findings)} findings):**
{findings_text}

**Your Analysis Process:**

Step 1: Identify which SINGLE pattern in the user's code most closely matches a finding
Step 2: Verify this pattern actually exists in BOTH user code and cited finding
Step 3: If verified, report ONLY that ONE vulnerability
Step 4: If no strong match (>80% confidence), report "No matching vulnerabilities detected"

**REQUIRED Output Format (MUST use exactly):**

## Vulnerability Analysis

### [Vulnerability Name] - [SEVERITY]
**Finding Reference:** [Finding N]
**Structural Match:** [Quote the specific lines from user code that match the finding pattern]
**Why This is Vulnerable:** [Explain how user code matches the vulnerable pattern in the finding]
**Impact:** [Based on the cited finding]
**Recommended Fix:** [Based on the cited finding]

CRITICAL: You MUST start with "## Vulnerability Analysis" header, then "### [Name] - [SEVERITY]" on the next line.
Do NOT use "##" for the vulnerability name - always use "###".

---

**CRITICAL RULES - VIOLATIONS WILL FAIL EVALUATION:**
- Report MAXIMUM ONE vulnerability (the most critical match)
- Do NOT list every vulnerability from the findings
- Do NOT report generic security advice
- Do NOT report vulnerabilities that aren't in the user's code
- ONLY report if you have >80% confidence the pattern matches
- MUST use the exact output format above (### for vulnerability name)

Begin your analysis (remember: ONE vulnerability maximum):
"""

        # Call LLM
        analysis = self.llm.invoke(prompt)

        return analysis

    def _format_patterns_for_llm(self, patterns: Dict) -> str:
        """Format structural patterns for LLM"""
        lines = []

        if patterns['external_calls']:
            lines.append(f"External Calls: {len(patterns['external_calls'])} found")
            for call in patterns['external_calls'][:3]:  # Show first 3
                lines.append(f"  - Line {call['line_number']}: {call['call_type']}")

        if patterns['state_changes']:
            lines.append(f"\nState Changes: {len(patterns['state_changes'])} found")
            for change in patterns['state_changes'][:3]:
                lines.append(f"  - Line {change['line_number']}: {change['variable']} ({change['operation']})")

        if patterns['ordering_patterns']:
            lines.append(f"\nOrdering Patterns:")
            for key, value in patterns['ordering_patterns'].items():
                if isinstance(value, bool) and value:
                    lines.append(f"  - {key}")
                elif isinstance(value, int):
                    lines.append(f"  - {key}: line {value}")

        if patterns['functions']:
            lines.append(f"\nFunctions: {len(patterns['functions'])} found")
            for func in patterns['functions']:
                modifiers_str = ', '.join(func['modifiers']) if func['modifiers'] else 'none'
                lines.append(f"  - {func['name']}: {func['visibility']}, modifiers: {modifiers_str}")

        if patterns['dangerous_operations']:
            lines.append(f"\nDangerous Operations: {', '.join(patterns['dangerous_operations'])}")

        if patterns.get('missing_access_control'):
            lines.append(f"\nMissing Access Control: {len(patterns['missing_access_control'])} function(s)")
            for func in patterns['missing_access_control']:
                lines.append(f"  - {func['function_name']} ({func['visibility']}) at line {func['line_number']} - no access modifier")

        return '\n'.join(lines) if lines else "No significant patterns detected"

    def _format_findings_for_llm(self, findings: List[Dict]) -> str:
        """Format retrieved findings for LLM"""
        formatted = []

        for i, finding in enumerate(findings, 1):
            similarity = finding.get('similarity_score', 'N/A')
            if similarity != 'N/A':
                similarity = f"{similarity:.1%}"

            formatted.append(f"""
[Finding {i}] {finding['title']} (Similarity: {similarity})
Firm: {finding['firm_name']} | Protocol: {finding['protocol_name']} | Impact: {finding['impact']}
Retrieved by: {finding['retrieval_strategy']}

{finding['content'][:800]}...

---
""")

        return '\n'.join(formatted)

    def export_to_json(self, result: Dict, output_file: str) -> None:
        """Export detection result to JSON file"""
        import json
        from datetime import datetime

        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_version': '1.0.0',
            'configuration': {
                'k_code': self.k_code,
                'k_text': self.k_text,
                'min_similarity': self.retriever.min_similarity,
                'model': self.model
            },
            'structural_patterns': {
                'external_calls_count': len(result['structural_patterns']['external_calls']),
                'state_changes_count': len(result['structural_patterns']['state_changes']),
                'functions_count': len(result['structural_patterns']['functions']),
                'dangerous_operations': result['structural_patterns']['dangerous_operations'],
                'ordering_patterns': result['structural_patterns']['ordering_patterns']
            },
            'retrieval_results': {
                'num_findings': len(result['retrieved_findings']),
                'findings': [
                    {
                        'title': f['title'],
                        'firm': f['firm_name'],
                        'protocol': f['protocol_name'],
                        'impact': f['impact'],
                        'similarity_score': f.get('similarity_score', 'N/A'),
                        'retrieval_strategy': f['retrieval_strategy']
                    }
                    for f in result['retrieved_findings']
                ]
            },
            'analysis': result['analysis'],
            'metadata': result['metadata']
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to {output_file}")

    def get_database_stats(self) -> Dict:
        """Get database coverage statistics"""
        import glob
        import json

        stats = {
            'total_findings': 0,
            'by_impact': {},
            'by_firm': {},
            'coverage': {}
        }

        # Count findings in dataset
        data_dir = self.retriever.code_db._persist_directory.replace('/code_db', '')
        data_dir = data_dir.replace('/databases', '')
        json_files = glob.glob(f"{data_dir}/../../sample-smart-contract-dataset/*.json")

        for filepath in json_files:
            try:
                with open(filepath, 'r') as f:
                    finding = json.load(f)

                stats['total_findings'] += 1

                # Count by impact
                impact = finding.get('impact', 'UNKNOWN')
                stats['by_impact'][impact] = stats['by_impact'].get(impact, 0) + 1

                # Count by firm
                firm = finding.get('firm_name', 'Unknown')
                stats['by_firm'][firm] = stats['by_firm'].get(firm, 0) + 1

            except:
                continue

        return stats

    def print_database_stats(self) -> None:
        """Print database coverage statistics"""
        print()
        print("=" * 80)
        print("DATABASE COVERAGE STATISTICS")
        print("=" * 80)

        stats = self.get_database_stats()

        print(f"\nTotal Findings in Database: {stats['total_findings']}")

        print(f"\nBy Impact Level:")
        for impact, count in sorted(stats['by_impact'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_findings']) * 100
            print(f"  {impact}: {count} ({percentage:.1f}%)")

        print(f"\nTop Audit Firms (by finding count):")
        top_firms = sorted(stats['by_firm'].items(), key=lambda x: x[1], reverse=True)[:5]
        for firm, count in top_firms:
            percentage = (count / stats['total_findings']) * 100
            print(f"  {firm}: {count} ({percentage:.1f}%)")

        print("=" * 80)
        print()


# Example usage
if __name__ == '__main__':
    test_code = """
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] = 0;
    }
    """

    print("Testing RAG Vulnerability Detector")
    print("=" * 80)
    print()
    print("Test Code:")
    print(test_code)
    print()

    # Note: This requires databases to be built first
    print("NOTE: This example requires databases to be built first.")
    print("Run: python databases/build_dual_databases.py")
    print()
    print("Then uncomment the code below to test:")
    print()

    # Uncomment to test after building databases:
    # detector = RAGVulnerabilityDetector()
    # result = detector.detect(test_code, verbose=True)
    #
    # print()
    # print("ANALYSIS:")
    # print("=" * 80)
    # print(result['analysis'])
