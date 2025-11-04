"""
Utility functions for the Smart Contract Security Assistant
"""

import json
from typing import Dict, List


def load_json_file(filepath: str) -> Dict:
    """Load a JSON file and return its contents"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_vulnerability_report(analysis: Dict) -> str:
    """Format vulnerability analysis into readable report"""
    # TODO: Implement nice formatting for analysis results
    return str(analysis)


def extract_code_snippets(text: str) -> List[str]:
    """Extract code snippets from markdown text"""
    # TODO: Extract code blocks from markdown
    snippets = []
    in_code_block = False
    current_snippet = []

    for line in text.split('\n'):
        if line.strip().startswith('```'):
            if in_code_block:
                snippets.append('\n'.join(current_snippet))
                current_snippet = []
            in_code_block = not in_code_block
        elif in_code_block:
            current_snippet.append(line)

    return snippets


def calculate_severity_score(vulnerabilities: List[Dict]) -> Dict[str, int]:
    """Calculate severity distribution from vulnerability list"""
    severity_count = {
        'CRITICAL': 0,
        'HIGH': 0,
        'MEDIUM': 0,
        'LOW': 0
    }

    for vuln in vulnerabilities:
        severity = vuln.get('severity', 'MEDIUM').upper()
        if severity in severity_count:
            severity_count[severity] += 1

    return severity_count


if __name__ == "__main__":
    # Test utilities
    print("Utility functions module")
