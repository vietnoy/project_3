#!/usr/bin/env python3
"""
Demo Detection with LLM Output Saving
======================================
Run detection on test cases and save full LLM analysis output
to readable files for demonstration purposes.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directories to path
sys.path.append('../..')
sys.path.append('..')

from evaluate_llm_output import GROUND_TRUTH_TESTS
from rag_detector import RAGVulnerabilityDetector


def run_demo_detection(
    test_indices: list = None,
    output_dir: str = './demo_outputs',
    code_db_path: str = '../databases/code_db',
    text_db_path: str = '../databases/text_db',
    verbose: bool = True
):
    """
    Run detection on specified test cases and save outputs

    Args:
        test_indices: List of test indices to run (1-indexed). If None, run all.
        output_dir: Directory to save outputs
        code_db_path: Path to code database
        text_db_path: Path to text database
        verbose: Print detailed output
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print()
    print("=" * 80)
    print("DEMO DETECTION WITH LLM OUTPUT SAVING")
    print("=" * 80)
    print()

    # Initialize detector
    print("Initializing RAG detector...")
    detector = RAGVulnerabilityDetector(
        code_db_path=code_db_path,
        text_db_path=text_db_path,
        llm_model='qwen2.5-coder:7b',
        k_code=4,
        k_text=4
    )
    print("✓ Detector initialized")
    print()

    # Determine which tests to run
    if test_indices is None:
        test_indices = list(range(1, len(GROUND_TRUTH_TESTS) + 1))

    tests_to_run = [(i, GROUND_TRUTH_TESTS[i-1]) for i in test_indices]

    print(f"Running detection on {len(tests_to_run)} test case(s)...")
    print()

    results = []

    for test_num, test in tests_to_run:
        print(f"[{test_num}/{len(GROUND_TRUTH_TESTS)}] {test.name}")
        print("-" * 80)

        # Run detection
        result = detector.detect(test.code, verbose=False)

        # Create output structure
        output_data = {
            'test_info': {
                'number': test_num,
                'name': test.name,
                'timestamp': datetime.now().isoformat(),
                'code': test.code.strip()
            },
            'expected': {
                'vulnerabilities': [
                    {
                        'type': vuln.type,
                        'severity': vuln.severity,
                        'description': vuln.description,
                        'must_contain_keywords': vuln.must_contain_keywords
                    }
                    for vuln in test.expected_vulnerabilities
                ]
            },
            'detection_result': {
                'structural_patterns': result['structural_patterns'],
                'retrieved_findings': [
                    {
                        'title': f['title'],
                        'finding_id': f.get('finding_id', 'N/A'),
                        'firm_name': f['firm_name'],
                        'protocol_name': f['protocol_name'],
                        'impact': f['impact'],
                        'similarity_score': f.get('similarity_score', 'N/A'),
                        'retrieval_strategy': f['retrieval_strategy']
                    }
                    for f in result['retrieved_findings']
                ],
                'llm_analysis': result['analysis'],
                'metadata': result['metadata']
            }
        }

        # Save to file
        filename = f"demo_{test_num:02d}_{test.name.lower().replace(' ', '_').replace('-', '_')}.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Also save LLM analysis as plain text for easy reading
        txt_filename = filename.replace('.json', '_llm_output.txt')
        txt_filepath = output_path / txt_filename

        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"TEST CASE: {test.name}\n")
            f.write("=" * 80 + "\n\n")
            f.write("CODE:\n")
            f.write("-" * 80 + "\n")
            f.write(test.code.strip() + "\n")
            f.write("-" * 80 + "\n\n")
            f.write("LLM ANALYSIS OUTPUT:\n")
            f.write("=" * 80 + "\n\n")
            f.write(result['analysis'] + "\n")

        print(f"✓ Saved JSON: {filename}")
        print(f"✓ Saved TXT:  {txt_filename}")

        if verbose:
            print()
            print("LLM Analysis Preview:")
            print("-" * 80)
            lines = result['analysis'].split('\n')
            for line in lines[:15]:  # Show first 15 lines
                print(line)
            if len(lines) > 15:
                print("... (see full output in file) ...")
            print("-" * 80)

        print()

        results.append(output_data)

    # Create summary index
    index_path = output_path / "00_demo_index.json"
    index = {
        'generated_at': datetime.now().isoformat(),
        'total_demos': len(results),
        'demos': [
            {
                'number': r['test_info']['number'],
                'name': r['test_info']['name'],
                'json_file': f"demo_{r['test_info']['number']:02d}_{r['test_info']['name'].lower().replace(' ', '_').replace('-', '_')}.json",
                'txt_file': f"demo_{r['test_info']['number']:02d}_{r['test_info']['name'].lower().replace(' ', '_').replace('-', '_')}_llm_output.txt",
                'num_findings_retrieved': len(r['detection_result']['retrieved_findings']),
                'has_analysis': bool(r['detection_result']['llm_analysis'])
            }
            for r in results
        ]
    }

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print("DEMO DETECTION COMPLETE")
    print("=" * 80)
    print(f"Outputs saved to: {output_path.absolute()}")
    print(f"Total files: {len(results) * 2 + 1} (JSON + TXT + index)")
    print()
    print("Files created:")
    print(f"  - 00_demo_index.json (index of all demos)")
    for r in results:
        num = r['test_info']['number']
        name_slug = r['test_info']['name'].lower().replace(' ', '_').replace('-', '_')
        print(f"  - demo_{num:02d}_{name_slug}.json")
        print(f"  - demo_{num:02d}_{name_slug}_llm_output.txt")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run demo detection and save LLM outputs')
    parser.add_argument('--tests', '-t', type=int, nargs='+',
                        help='Test numbers to run (e.g., 1 3 5). If not specified, run all.')
    parser.add_argument('--output', '-o', default='./demo_outputs',
                        help='Output directory (default: ./demo_outputs)')
    parser.add_argument('--code-db', default='../databases/code_db',
                        help='Path to code database')
    parser.add_argument('--text-db', default='../databases/text_db',
                        help='Path to text database')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    run_demo_detection(
        test_indices=args.tests,
        output_dir=args.output,
        code_db_path=args.code_db,
        text_db_path=args.text_db,
        verbose=not args.quiet
    )
