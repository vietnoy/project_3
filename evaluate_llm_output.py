#!/usr/bin/env python3
"""
LLM Output Evaluation Framework
================================
Properly evaluates if the LLM correctly identifies vulnerabilities
NOT just if features are extracted!

Metrics:
- Precision: % of reported vulnerabilities that are real
- Recall: % of real vulnerabilities detected
- F1 Score: Harmonic mean
- Citation Accuracy: Are [Finding N] references relevant?
"""
import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class GroundTruthVulnerability:
    """Ground truth for a single vulnerability"""
    type: str  # e.g., "reentrancy", "access-control"
    severity: str  # HIGH, MEDIUM, LOW
    must_contain_keywords: List[str]  # Must appear in LLM explanation
    description: str  # What the vulnerability is


@dataclass
class GroundTruthTest:
    """Complete test case with expected results"""
    name: str
    code: str
    expected_vulnerabilities: List[GroundTruthVulnerability]
    should_have_zero_vulns: bool = False  # True if code is safe


# Ground Truth Test Cases
GROUND_TRUTH_TESTS = [
    GroundTruthTest(
        name="Reentrancy in withdraw function",
        code="""
        function withdraw() public {
            uint256 amount = balances[msg.sender];
            (bool success,) = msg.sender.call{value: amount}("");
            require(success);
            balances[msg.sender] = 0;
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="reentrancy",
                severity="HIGH",
                must_contain_keywords=["external call", "state", "reentrancy"],
                description="External call before state update"
            )
        ]
    ),

    GroundTruthTest(
        name="Missing access control on critical function",
        code="""
        function setOwner(address newOwner) public {
            owner = newOwner;
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="access-control",
                severity="HIGH",
                must_contain_keywords=["access", "modifier", "public"],
                description="No access control on critical function"
            )
        ]
    ),

    GroundTruthTest(
        name="Unchecked return value of external call",
        code="""
        function executeTransaction(address target, bytes memory data) public {
            target.call(data);
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="unchecked-call",
                severity="MEDIUM",
                must_contain_keywords=["return value", "unchecked", "call"],
                description="Return value of call not checked"
            )
        ]
    ),

    GroundTruthTest(
        name="Timestamp dependence for critical logic",
        code="""
        function claimReward() public {
            require(block.timestamp % 100 == 0, "Not eligible");
            rewards[msg.sender] += 100;
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="timestamp-dependence",
                severity="MEDIUM",
                must_contain_keywords=["timestamp", "manipulat", "miner"],
                description="Using block.timestamp for critical logic"
            )
        ]
    ),

    GroundTruthTest(
        name="tx.origin for authentication",
        code="""
        function withdraw() public {
            require(tx.origin == owner);
            payable(msg.sender).transfer(address(this).balance);
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="tx-origin",
                severity="HIGH",
                must_contain_keywords=["tx.origin", "phishing", "authentication"],
                description="Using tx.origin for authentication"
            )
        ]
    ),

    GroundTruthTest(
        name="Arbitrary delegatecall vulnerability",
        code="""
        function execute(address target, bytes memory data) public {
            target.delegatecall(data);
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="delegatecall",
                severity="CRITICAL",
                must_contain_keywords=["delegatecall", "arbitrary", "context"],
                description="User-controlled delegatecall target"
            )
        ]
    ),

    GroundTruthTest(
        name="Unbounded loop DoS vulnerability",
        code="""
        function distributeRewards() public {
            for (uint i = 0; i < users.length; i++) {
                users[i].transfer(100);
            }
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="dos",
                severity="MEDIUM",
                must_contain_keywords=["loop", "unbounded", "gas", "dos"],
                description="Unbounded loop over dynamic array"
            )
        ]
    ),

    GroundTruthTest(
        name="Off-by-one error in loop",
        code="""
        for (uint256 i = stakes.length; i > 0; i--) {
            Stake storage stake = stakes[i];
            processStake(stake);
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="off-by-one",
                severity="HIGH",
                must_contain_keywords=["off-by-one", "index", "array", "bounds"],
                description="Array access out of bounds"
            )
        ]
    ),

    GroundTruthTest(
        name="Missing zero address check",
        code="""
        function setController(address _controller) external {
            controller = _controller;
        }
        """,
        expected_vulnerabilities=[
            GroundTruthVulnerability(
                type="missing-validation",
                severity="LOW",
                must_contain_keywords=["zero address", "validation", "check"],
                description="No zero address validation"
            )
        ]
    ),

    GroundTruthTest(
        name="Safe code - properly implemented withdraw",
        code="""
        function withdraw() external onlyOwner nonReentrant {
            uint256 amount = balances[msg.sender];
            balances[msg.sender] = 0;
            (bool success,) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
        }
        """,
        expected_vulnerabilities=[],
        should_have_zero_vulns=True
    ),
]


class LLMOutputEvaluator:
    """Evaluates LLM vulnerability analysis output"""

    def extract_vulnerabilities_from_text(self, llm_output: str) -> List[Dict]:
        """
        Parse LLM output to extract detected vulnerabilities

        Returns list of: {'type': str, 'severity': str, 'explanation': str, 'citations': List[str]}
        """
        vulnerabilities = []

        # Split by vulnerability sections (look for ### headers)
        sections = re.split(r'###\s+', llm_output)

        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n')
            if not lines:
                continue

            # Parse header: "Vulnerability Name - SEVERITY"
            header = lines[0].strip()

            # Extract vulnerability type
            vuln_type = header.lower().split('-')[0].strip()

            # Extract severity
            severity_match = re.search(r'\b(CRITICAL|HIGH|MEDIUM|LOW)\b', header, re.IGNORECASE)
            severity = severity_match.group(1).upper() if severity_match else "UNKNOWN"

            # Extract explanation (all content)
            explanation = '\n'.join(lines[1:])

            # Extract citations
            citations = re.findall(r'\[Finding \d+\]', explanation)

            vulnerabilities.append({
                'type': vuln_type,
                'severity': severity,
                'explanation': explanation,
                'citations': citations
            })

        return vulnerabilities

    def match_vulnerability(
        self,
        detected: Dict,
        expected: GroundTruthVulnerability
    ) -> Tuple[bool, float]:
        """
        Check if detected vulnerability matches expected one

        Returns: (is_match, confidence_score)
        """
        # Check type match (fuzzy)
        type_keywords = expected.type.lower().split('-')
        detected_text = (detected['type'] + ' ' + detected['explanation']).lower()

        type_match = any(keyword in detected_text for keyword in type_keywords)

        if not type_match:
            return False, 0.0

        # Check if required keywords appear in explanation
        explanation_lower = detected['explanation'].lower()
        keyword_matches = sum(
            1 for keyword in expected.must_contain_keywords
            if keyword.lower() in explanation_lower
        )

        keyword_score = keyword_matches / len(expected.must_contain_keywords)

        # Check severity (allow one level difference)
        severity_match = self._severity_close(detected['severity'], expected.severity)

        # Check citations exist
        has_citations = len(detected['citations']) > 0

        # Calculate confidence
        confidence = (
            0.4 * (1.0 if type_match else 0.0) +
            0.3 * keyword_score +
            0.2 * (1.0 if severity_match else 0.0) +
            0.1 * (1.0 if has_citations else 0.0)
        )

        return confidence > 0.6, confidence

    def _severity_close(self, detected: str, expected: str) -> bool:
        """Check if severities are close (within one level)"""
        severity_levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3, 'UNKNOWN': 1}
        detected_level = severity_levels.get(detected, 1)
        expected_level = severity_levels.get(expected, 1)
        return abs(detected_level - expected_level) <= 1

    def evaluate_test_case(
        self,
        test: GroundTruthTest,
        llm_output: str
    ) -> Dict:
        """
        Evaluate LLM output against ground truth for one test case

        Returns metrics for this test case
        """
        detected_vulns = self.extract_vulnerabilities_from_text(llm_output)

        if test.should_have_zero_vulns:
            # Safe code - should detect nothing
            false_positives = len(detected_vulns)
            return {
                'test_name': test.name,
                'expected_vulns': 0,
                'detected_vulns': len(detected_vulns),
                'true_positives': 0,
                'false_positives': false_positives,
                'false_negatives': 0,
                'precision': 1.0 if false_positives == 0 else 0.0,
                'recall': 1.0,  # No vulns to miss
                'f1': 1.0 if false_positives == 0 else 0.0,
                'passed': false_positives == 0
            }

        # Match detected to expected
        matched_expected = set()
        matched_detected = set()

        for i, expected in enumerate(test.expected_vulnerabilities):
            for j, detected in enumerate(detected_vulns):
                is_match, confidence = self.match_vulnerability(detected, expected)
                if is_match and i not in matched_expected:
                    matched_expected.add(i)
                    matched_detected.add(j)
                    break

        true_positives = len(matched_expected)
        false_negatives = len(test.expected_vulnerabilities) - true_positives
        false_positives = len(detected_vulns) - len(matched_detected)

        precision = true_positives / len(detected_vulns) if detected_vulns else 0.0
        recall = true_positives / len(test.expected_vulnerabilities) if test.expected_vulnerabilities else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            'test_name': test.name,
            'expected_vulns': len(test.expected_vulnerabilities),
            'detected_vulns': len(detected_vulns),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'passed': f1 >= 0.7  # 70% threshold
        }

    def evaluate_system(
        self,
        detector_func,
        tests: List[GroundTruthTest] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a vulnerability detector system on all test cases

        Args:
            detector_func: Function that takes code and returns {'analysis': str, ...}
            tests: List of test cases (defaults to GROUND_TRUTH_TESTS)
            verbose: Print detailed results

        Returns:
            Aggregate metrics across all tests
        """
        if tests is None:
            tests = GROUND_TRUTH_TESTS

        results = []

        if verbose:
            print("\n" + "=" * 80)
            print("LLM OUTPUT EVALUATION")
            print("=" * 80)

        for i, test in enumerate(tests, 1):
            if verbose:
                print(f"\n[{i}/{len(tests)}] {test.name}")
                print("-" * 80)

            try:
                # Run detector
                result = detector_func(test.code)
                llm_output = result.get('analysis', '')

                if verbose:
                    print(f"Expected: {len(test.expected_vulnerabilities)} vulnerabilities")
                    print(f"LLM Output (first 200 chars): {llm_output[:200]}...")

                # Evaluate
                metrics = self.evaluate_test_case(test, llm_output)
                results.append(metrics)

                if verbose:
                    print(f"Detected: {metrics['detected_vulns']} vulnerabilities")
                    print(f"TP: {metrics['true_positives']}, "
                          f"FP: {metrics['false_positives']}, "
                          f"FN: {metrics['false_negatives']}")
                    print(f"Precision: {metrics['precision']:.2%}, "
                          f"Recall: {metrics['recall']:.2%}, "
                          f"F1: {metrics['f1']:.2%}")
                    print(f"Result: {'✅ PASSED' if metrics['passed'] else '❌ FAILED'}")

            except Exception as e:
                if verbose:
                    print(f"❌ ERROR: {e}")
                results.append({
                    'test_name': test.name,
                    'error': str(e),
                    'passed': False
                })

        # Aggregate metrics
        passed = sum(1 for r in results if r.get('passed', False))
        total = len(results)

        valid_results = [r for r in results if 'error' not in r]

        avg_precision = sum(r['precision'] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_recall = sum(r['recall'] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_f1 = sum(r['f1'] for r in valid_results) / len(valid_results) if valid_results else 0

        total_tp = sum(r['true_positives'] for r in valid_results)
        total_fp = sum(r['false_positives'] for r in valid_results)
        total_fn = sum(r['false_negatives'] for r in valid_results)

        if verbose:
            print("\n" + "=" * 80)
            print("AGGREGATE RESULTS")
            print("=" * 80)
            print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)")
            print(f"\nAverage Metrics:")
            print(f"  Precision: {avg_precision:.2%}")
            print(f"  Recall: {avg_recall:.2%}")
            print(f"  F1 Score: {avg_f1:.2%}")
            print(f"\nTotal Counts:")
            print(f"  True Positives: {total_tp}")
            print(f"  False Positives: {total_fp}")
            print(f"  False Negatives: {total_fn}")
            print("=" * 80)

        return {
            'tests_passed': passed,
            'tests_total': total,
            'pass_rate': passed / total if total > 0 else 0,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1,
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'individual_results': results
        }


if __name__ == "__main__":
    print("LLM Output Evaluation Framework")
    print("=" * 80)
    print(f"\nGround Truth Test Cases: {len(GROUND_TRUTH_TESTS)}")
    print("\nTest Categories:")
    print("  - Reentrancy")
    print("  - Access Control")
    print("  - Unchecked Calls")
    print("  - Timestamp Dependence")
    print("  - tx.origin")
    print("  - Delegatecall")
    print("  - DoS (Unbounded Loop)")
    print("  - Off-by-One")
    print("  - Missing Validation")
    print("  - Safe Code (No Vulnerabilities)")
    print("\n" + "=" * 80)
    print("\nTo use this framework:")
    print("  from evaluate_llm_output import LLMOutputEvaluator, GROUND_TRUTH_TESTS")
    print("  evaluator = LLMOutputEvaluator()")
    print("  metrics = evaluator.evaluate_system(your_detector_function)")
