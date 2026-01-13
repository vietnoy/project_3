#!/usr/bin/env python3
"""
Export Test Cases to Readable JSON Files
=========================================
This script exports all test cases from evaluate_llm_output.py
into individual JSON files for easy viewing.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append('../..')

from evaluate_llm_output import GROUND_TRUTH_TESTS


def export_test_cases(output_dir: str = './test_cases_export'):
    """
    Export all test cases to JSON files

    Args:
        output_dir: Directory to save test case files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Exporting {len(GROUND_TRUTH_TESTS)} test cases to {output_dir}/")
    print()

    for i, test in enumerate(GROUND_TRUTH_TESTS, 1):
        # Create filename from test name
        filename = test.name.lower().replace(' ', '_').replace('-', '_')
        filepath = output_path / f"test_{i:02d}_{filename}.json"

        # Convert to dictionary
        test_dict = {
            'test_number': i,
            'name': test.name,
            'code': test.code.strip(),
            'should_have_zero_vulns': test.should_have_zero_vulns,
            'expected_vulnerabilities': [
                {
                    'type': vuln.type,
                    'severity': vuln.severity,
                    'must_contain_keywords': vuln.must_contain_keywords,
                    'description': vuln.description
                }
                for vuln in test.expected_vulnerabilities
            ]
        }

        # Save to JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(test_dict, f, indent=2, ensure_ascii=False)

        print(f"✓ Exported: {filepath.name}")

    # Create index file
    index_path = output_path / "00_index.json"
    index = {
        'total_tests': len(GROUND_TRUTH_TESTS),
        'tests': [
            {
                'number': i,
                'name': test.name,
                'file': f"test_{i:02d}_{test.name.lower().replace(' ', '_').replace('-', '_')}.json",
                'has_vulnerabilities': not test.should_have_zero_vulns,
                'num_expected_vulns': len(test.expected_vulnerabilities)
            }
            for i, test in enumerate(GROUND_TRUTH_TESTS, 1)
        ]
    }

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print()
    print(f"✓ Created index: {index_path.name}")
    print()
    print("=" * 60)
    print("EXPORT COMPLETE")
    print("=" * 60)
    print(f"Total files created: {len(GROUND_TRUTH_TESTS) + 1}")
    print(f"Location: {output_path.absolute()}")
    print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Export test cases to JSON files')
    parser.add_argument('--output', '-o', default='./test_cases_export',
                        help='Output directory (default: ./test_cases_export)')

    args = parser.parse_args()

    export_test_cases(args.output)
