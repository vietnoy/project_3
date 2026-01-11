#!/usr/bin/env python3
"""
RAG System Evaluation with LLM Output Metrics
=============================================

Properly evaluates the RAG-heavy system by measuring:
- Precision: % of reported vulnerabilities that are real
- Recall: % of real vulnerabilities detected
- F1 Score: Overall accuracy
- Citation Quality: Are [Finding N] references relevant?

NOT just feature extraction - actual LLM output quality!
"""
import sys
sys.path.append('../..')  # Add parent directory to path

from typing import Dict
import time

# Import RAG detector
sys.path.append('..')
from rag_detector import RAGVulnerabilityDetector

# Import evaluation framework
from evaluate_llm_output import LLMOutputEvaluator, GROUND_TRUTH_TESTS


def evaluate_rag_system(
    code_db_path: str = '../databases/code_db',
    text_db_path: str = '../databases/text_db',
    llm_model: str = 'qwen2.5-coder:7b',
    k_code: int = 4,
    k_text: int = 4
):
    """
    Evaluate RAG system with proper LLM output metrics

    Args:
        code_db_path: Path to code database
        text_db_path: Path to text database
        llm_model: Ollama model to use
        k_code: Number of code similarities to retrieve (default: 3)
        k_text: Number of text patterns to retrieve (default: 3)
    """
    print()
    print("=" * 80)
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)
    print()
    print("Evaluation Framework: LLM Output Quality Assessment")
    print(f"Test Cases: {len(GROUND_TRUTH_TESTS)}")
    print(f"Metrics: Precision, Recall, F1 Score, Citation Quality")
    print()
    print("Configuration:")
    print(f"  Code DB: {code_db_path}")
    print(f"  Text DB: {text_db_path}")
    print(f"  LLM: {llm_model}")
    print(f"  k_code: {k_code}")
    print(f"  k_text: {k_text}")
    print("=" * 80)
    print()

    # Initialize detector
    print("Initializing RAG detector...")
    print(f"  Loading code database from: {code_db_path}")
    print(f"  Loading text database from: {text_db_path}")
    print(f"  Using LLM model: {llm_model}")
    print()

    detector = RAGVulnerabilityDetector(
        code_db_path=code_db_path,
        text_db_path=text_db_path,
        llm_model=llm_model,
        k_code=k_code,
        k_text=k_text
    )

    print("Detector initialized successfully!")
    print()

    # Create detector function for evaluator
    def detector_function(code: str) -> Dict:
        """Wrapper function for evaluation"""
        result = detector.detect(code, verbose=False)
        return {
            'analysis': result['analysis'],
            'structural_patterns': result['structural_patterns'],
            'retrieved_findings': result['retrieved_findings'],
            'metadata': result['metadata']
        }

    # Run evaluation
    evaluator = LLMOutputEvaluator()

    print()
    print("Running evaluation on ground truth test cases...")
    print("This will take time - LLM needs to analyze each test case.")
    print()

    start_time = time.time()

    metrics = evaluator.evaluate_system(
        detector_func=detector_function,
        tests=GROUND_TRUTH_TESTS,
        verbose=True
    )

    elapsed_time = time.time() - start_time

    # Print detailed results
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print()
    print(f"Tests Passed: {metrics['tests_passed']}/{metrics['tests_total']} ({metrics['pass_rate']*100:.1f}%)")
    print()
    print("Aggregate Metrics:")
    print(f"  Precision: {metrics['avg_precision']:.2%}")
    print(f"  Recall: {metrics['avg_recall']:.2%}")
    print(f"  F1 Score: {metrics['avg_f1']:.2%}")
    print()
    print("Detection Counts:")
    print(f"  True Positives: {metrics['total_true_positives']}")
    print(f"  False Positives: {metrics['total_false_positives']}")
    print(f"  False Negatives: {metrics['total_false_negatives']}")
    print()
    print(f"Evaluation Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print("=" * 80)
    print()

    # Per-test breakdown
    print("Per-Test Results:")
    print("-" * 80)
    for result in metrics['individual_results']:
        if 'error' in result:
            print(f"  {result['test_name']}: ERROR - {result['error']}")
        else:
            status = "PASS" if result['passed'] else "FAIL"
            print(f"  {result['test_name']}: {status}")
            print(f"    Precision: {result['precision']:.2%}, Recall: {result['recall']:.2%}, F1: {result['f1']:.2%}")

    print()
    print("=" * 80)

    # Interpretation
    print()
    print("INTERPRETATION:")
    print("=" * 80)

    if metrics['avg_f1'] >= 0.8:
        print("EXCELLENT performance (F1 >= 80%)")
        print("The RAG-heavy system is working very well!")
    elif metrics['avg_f1'] >= 0.7:
        print("GOOD performance (F1 >= 70%)")
        print("System meets baseline requirements.")
    elif metrics['avg_f1'] >= 0.6:
        print("MODERATE performance (F1 >= 60%)")
        print("System shows promise but needs tuning.")
    else:
        print("NEEDS IMPROVEMENT (F1 < 60%)")
        print("Consider:")
        print("  - Increasing k_code and k_text values")
        print("  - Improving structural pattern extraction")
        print("  - Tuning LLM prompt for better discovery")

    print()
    print("=" * 80)
    print()

    return metrics


def main():
    """Main evaluation entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RAG vulnerability detector')
    parser.add_argument('--code-db', default='../databases/code_db',
                        help='Path to code database')
    parser.add_argument('--text-db', default='../databases/text_db',
                        help='Path to text database')
    parser.add_argument('--model', default='qwen2.5-coder:7b',
                        help='Ollama model name')
    parser.add_argument('--k-code', type=int, default=4,
                        help='Number of code similarities to retrieve (default: 3)')
    parser.add_argument('--k-text', type=int, default=4,
                        help='Number of text patterns to retrieve (default: 3)')

    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_rag_system(
        code_db_path=args.code_db,
        text_db_path=args.text_db,
        llm_model=args.model,
        k_code=args.k_code,
        k_text=args.k_text
    )

    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        # Convert non-serializable objects
        results_json = {
            'pass_rate': metrics['pass_rate'],
            'precision': metrics['avg_precision'],
            'recall': metrics['avg_recall'],
            'f1': metrics['avg_f1'],
            'tests_passed': metrics['tests_passed'],
            'tests_total': metrics['tests_total'],
            'true_positives': metrics['total_true_positives'],
            'false_positives': metrics['total_false_positives'],
            'false_negatives': metrics['total_false_negatives']
        }
        json.dump(results_json, f, indent=2)

    print(f"Results saved to evaluation_results.json")
    print()


if __name__ == '__main__':
    main()
