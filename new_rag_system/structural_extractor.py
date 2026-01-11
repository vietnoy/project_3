"""
Structural Pattern Extractor for RAG-Heavy System
=================================================

Extracts ONLY generic structural patterns from Solidity code.
Does NOT classify vulnerabilities - that's the LLM's job using retrieved examples.

Key principle: Extract WHAT IS IN THE CODE, not WHAT IT MEANS.

Returns:
- external_calls: Where and what type of external calls
- state_changes: Where variables are modified
- control_structures: Loops, conditionals, their characteristics
- access_modifiers: Function visibility
- dangerous_operations: Low-level operations (call, delegatecall, etc.)

NO vulnerability names! Let the RAG + LLM discover those.
"""
import re
from typing import Dict, List, Set
from dataclasses import dataclass, asdict


@dataclass
class ExternalCall:
    """Represents an external call in the code"""
    line_number: int
    call_type: str  # 'call', 'delegatecall', 'transfer', 'send'
    has_value: bool
    target: str  # Variable name if identifiable


@dataclass
class StateChange:
    """Represents a state variable modification"""
    line_number: int
    variable: str
    operation: str  # 'assignment', 'increment', 'decrement'


@dataclass
class ControlStructure:
    """Represents a control structure (loop, conditional)"""
    type: str  # 'for', 'while', 'if', 'require', 'assert'
    line_number: int
    condition: str  # The condition expression


@dataclass
class FunctionInfo:
    """Represents function characteristics"""
    name: str
    visibility: str  # 'public', 'external', 'internal', 'private'
    modifiers: List[str]
    has_payable: bool
    line_number: int


class StructuralPatternExtractor:
    """
    Extracts structural patterns without vulnerability classification.

    Philosophy: Describe the CODE STRUCTURE, let RAG find similar vulnerable patterns.
    """

    def extract_patterns(self, code: str) -> Dict:
        """
        Extract all structural patterns from code

        Returns dict with:
        - external_calls: List of external call information
        - state_changes: List of state modifications
        - control_structures: List of loops/conditionals
        - functions: List of function information
        - dangerous_operations: List of risky operations (for filtering retrieval)
        - ordering: Important pattern - what comes before what
        """
        lines = code.split('\n')

        functions = self._find_functions(lines)

        return {
            'external_calls': self._find_external_calls(lines),
            'state_changes': self._find_state_changes(lines),
            'control_structures': self._find_control_structures(lines),
            'functions': functions,
            'dangerous_operations': self._find_dangerous_operations(lines),
            'ordering_patterns': self._find_ordering_patterns(lines),
            'missing_access_control': self._find_missing_access_control(functions),
            'keywords': self._extract_keywords(code)
        }

    def _find_external_calls(self, lines: List[str]) -> List[Dict]:
        """Find all external calls with their characteristics"""
        external_calls = []

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Low-level call
            if '.call{' in line or '.call(' in line:
                external_calls.append({
                    'line_number': line_num,
                    'call_type': 'call',
                    'has_value': 'value:' in line_lower or '{value' in line_lower,
                    'raw': line.strip()
                })

            # Delegatecall
            elif '.delegatecall(' in line_lower or '.delegatecall{' in line_lower:
                external_calls.append({
                    'line_number': line_num,
                    'call_type': 'delegatecall',
                    'has_value': False,
                    'raw': line.strip()
                })

            # Transfer/send
            elif '.transfer(' in line_lower:
                external_calls.append({
                    'line_number': line_num,
                    'call_type': 'transfer',
                    'has_value': True,
                    'raw': line.strip()
                })

            elif '.send(' in line_lower:
                external_calls.append({
                    'line_number': line_num,
                    'call_type': 'send',
                    'has_value': True,
                    'raw': line.strip()
                })

            # External function calls (any .)
            elif re.search(r'\w+\.\w+\(', line):
                # Only if it looks like an external call, not just method call
                if not any(keyword in line_lower for keyword in ['this.', 'super.', 'library.']):
                    external_calls.append({
                        'line_number': line_num,
                        'call_type': 'external_function',
                        'has_value': False,
                        'raw': line.strip()
                    })

        return external_calls

    def _find_state_changes(self, lines: List[str]) -> List[Dict]:
        """Find state variable modifications"""
        state_changes = []

        for line_num, line in enumerate(lines, 1):
            # Simple assignment pattern: var = value
            if '=' in line and not '==' in line and not '!=' in line:
                # Extract variable name (left side of =)
                match = re.search(r'(\w+(?:\[\w+\])?)\s*=', line)
                if match:
                    var_name = match.group(1)

                    # Determine operation type
                    operation = 'assignment'
                    if '+=' in line:
                        operation = 'increment'
                    elif '-=' in line:
                        operation = 'decrement'
                    elif '*=' in line or '/=' in line:
                        operation = 'arithmetic'

                    state_changes.append({
                        'line_number': line_num,
                        'variable': var_name,
                        'operation': operation,
                        'raw': line.strip()
                    })

        return state_changes

    def _find_control_structures(self, lines: List[str]) -> List[Dict]:
        """Find loops and conditionals"""
        structures = []

        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            line_lower = line_lower = line_stripped.lower()

            # For loops
            if line_stripped.startswith('for') and '(' in line:
                match = re.search(r'for\s*\((.*?)\)', line)
                condition = match.group(1) if match else 'unknown'

                structures.append({
                    'type': 'for_loop',
                    'line_number': line_num,
                    'condition': condition,
                    'raw': line.strip()
                })

            # While loops
            elif line_stripped.startswith('while'):
                match = re.search(r'while\s*\((.*?)\)', line)
                condition = match.group(1) if match else 'unknown'

                structures.append({
                    'type': 'while_loop',
                    'line_number': line_num,
                    'condition': condition,
                    'raw': line.strip()
                })

            # If conditionals
            elif line_stripped.startswith('if'):
                match = re.search(r'if\s*\((.*?)\)', line)
                condition = match.group(1) if match else 'unknown'

                structures.append({
                    'type': 'conditional',
                    'line_number': line_num,
                    'condition': condition,
                    'raw': line.strip()
                })

            # Require statements
            elif 'require(' in line_lower:
                match = re.search(r'require\s*\((.*?)\)', line)
                condition = match.group(1) if match else 'unknown'

                structures.append({
                    'type': 'require',
                    'line_number': line_num,
                    'condition': condition,
                    'raw': line.strip()
                })

            # Assert statements
            elif 'assert(' in line_lower:
                match = re.search(r'assert\s*\((.*?)\)', line)
                condition = match.group(1) if match else 'unknown'

                structures.append({
                    'type': 'assert',
                    'line_number': line_num,
                    'condition': condition,
                    'raw': line.strip()
                })

        return structures

    def _find_functions(self, lines: List[str]) -> List[Dict]:
        """Find function declarations and their characteristics"""
        functions = []

        for line_num, line in enumerate(lines, 1):
            if 'function ' in line:
                # Extract function name
                match = re.search(r'function\s+(\w+)', line)
                if not match:
                    continue

                func_name = match.group(1)

                # Extract visibility
                visibility = 'internal'  # default
                if 'public' in line:
                    visibility = 'public'
                elif 'external' in line:
                    visibility = 'external'
                elif 'private' in line:
                    visibility = 'private'

                # Extract modifiers
                modifiers = []
                if 'view' in line:
                    modifiers.append('view')
                if 'pure' in line:
                    modifiers.append('pure')
                if 'payable' in line:
                    modifiers.append('payable')

                # Check for custom modifiers
                modifier_pattern = r'\)\s+(\w+(?:\([^)]*\))?)\s*(?:{|returns)'
                custom_mods = re.findall(modifier_pattern, line)
                modifiers.extend([m.split('(')[0] for m in custom_mods if m not in ['view', 'pure', 'payable', 'returns']])

                functions.append({
                    'name': func_name,
                    'visibility': visibility,
                    'modifiers': modifiers,
                    'has_payable': 'payable' in line,
                    'line_number': line_num,
                    'raw': line.strip()
                })

        return functions

    def _find_missing_access_control(self, functions: List[Dict]) -> List[Dict]:
        """
        Detect functions that SHOULD have access control but don't.

        Critical function patterns that need access control:
        - Owner/admin functions: setOwner, transferOwnership, updateAdmin
        - Financial functions: withdraw, transfer, mint, burn
        - Configuration functions: setPrice, updateFee, changeConfig
        - Public/external functions modifying state with no modifiers

        Returns list of functions missing access control.
        """
        missing_ac = []

        # Critical function name patterns (case-insensitive)
        critical_patterns = [
            'setowner', 'transferownership', 'changeowner', 'updateowner',
            'setadmin', 'updateadmin',
            'withdraw', 'withdrawfunds', 'emergencywithdraw',
            'mint', 'burn',
            'setprice', 'updateprice', 'changeprice',
            'setfee', 'updatefee', 'changefee',
            'pause', 'unpause',
            'upgrade', 'migrate',
            'destroy', 'selfdestruct'
        ]

        for func in functions:
            func_name_lower = func['name'].lower()

            # Check if it's a critical function
            is_critical = any(pattern in func_name_lower for pattern in critical_patterns)

            # Check if it's public/external
            is_exposed = func['visibility'] in ['public', 'external']

            # Check if it has access control modifiers
            # Common access control modifier names
            has_access_control = any(
                mod.lower() in ['onlyowner', 'onlyadmin', 'onlyauthorized', 'requiresauth']
                for mod in func['modifiers']
            )

            # Flag if critical AND exposed AND no access control
            if is_critical and is_exposed and not has_access_control:
                missing_ac.append({
                    'function_name': func['name'],
                    'line_number': func['line_number'],
                    'visibility': func['visibility'],
                    'modifiers': func['modifiers'],
                    'reason': 'critical_function_without_access_control'
                })

        return missing_ac

    def _find_dangerous_operations(self, lines: List[str]) -> List[str]:
        """
        Find potentially dangerous operations (for retrieval filtering).
        NOT vulnerability classification - just factual observations.
        """
        operations = set()
        code = '\n'.join(lines).lower()

        # Low-level operations
        if '.call(' in code or '.call{' in code:
            operations.add('low_level_call')
        if 'delegatecall' in code:
            operations.add('delegatecall_operation')
        if 'selfdestruct' in code:
            operations.add('selfdestruct_operation')

        # Block properties
        if 'block.timestamp' in code:
            operations.add('timestamp_usage')
        if 'block.number' in code:
            operations.add('block_number_usage')
        if 'blockhash' in code:
            operations.add('blockhash_usage')

        # Transaction properties
        if 'tx.origin' in code:
            operations.add('tx_origin_usage')
        if 'msg.sender' in code:
            operations.add('msg_sender_usage')

        # Array operations
        if '.length' in code and 'for' in code:
            operations.add('array_iteration')
        if '.push(' in code or '.pop(' in code:
            operations.add('dynamic_array_modification')

        return list(operations)

    def _find_ordering_patterns(self, lines: List[str]) -> Dict:
        """
        Find important ordering patterns in code.
        Example: external call on line 5, state change on line 7

        Returns orderings like:
        - call_before_state_change: True/False
        - state_change_before_call: True/False
        """
        # Get line numbers
        call_lines = set()
        state_lines = set()

        for line_num, line in enumerate(lines, 1):
            line_lower = line.lower()

            # Check for calls
            if any(pattern in line_lower for pattern in ['.call(', '.call{', 'delegatecall', '.transfer(', '.send(']):
                call_lines.add(line_num)

            # Check for state changes
            if '=' in line and not '==' in line and not '!=' in line:
                state_lines.add(line_num)

        # Analyze ordering
        ordering = {}

        if call_lines and state_lines:
            earliest_call = min(call_lines)
            latest_call = max(call_lines)
            earliest_state = min(state_lines)
            latest_state = max(state_lines)

            ordering['has_call_before_state_change'] = earliest_call < latest_state
            ordering['has_state_change_before_call'] = earliest_state < latest_call
            ordering['call_and_state_interleaved'] = (earliest_call < latest_state) and (earliest_state < latest_call)
            ordering['earliest_call_line'] = earliest_call
            ordering['earliest_state_line'] = earliest_state

        return ordering

    def _extract_keywords(self, code: str) -> List[str]:
        """Extract simple keywords for retrieval fallback"""
        keywords = set()
        code_lower = code.lower()

        # Function names that hint at purpose
        function_names = re.findall(r'function\s+(\w+)', code_lower)
        keywords.update(function_names[:5])  # Limit to avoid noise

        # Common patterns
        keyword_patterns = {
            'withdraw', 'deposit', 'transfer', 'send', 'mint', 'burn',
            'approve', 'owner', 'admin', 'pause', 'upgrade', 'initialize',
            'balance', 'amount', 'value', 'price', 'reward', 'stake'
        }

        for keyword in keyword_patterns:
            if keyword in code_lower:
                keywords.add(keyword)

        return list(keywords)

    def format_for_retrieval(self, patterns: Dict) -> str:
        """
        Format extracted patterns into a text query for retrieval.

        This creates a description of the code structure that can be
        matched against similar vulnerable code in the database.
        """
        query_parts = []

        # External calls
        if patterns['external_calls']:
            call_types = [call['call_type'] for call in patterns['external_calls']]
            query_parts.append(f"external calls: {', '.join(set(call_types))}")

        # State changes
        if patterns['state_changes']:
            query_parts.append(f"{len(patterns['state_changes'])} state modifications")

        # Control structures
        if patterns['control_structures']:
            struct_types = [s['type'] for s in patterns['control_structures']]
            query_parts.append(f"control structures: {', '.join(set(struct_types))}")

        # Functions
        if patterns['functions']:
            visibility = [f['visibility'] for f in patterns['functions']]
            query_parts.append(f"functions: {', '.join(set(visibility))}")

        # Dangerous operations
        if patterns['dangerous_operations']:
            query_parts.append(f"operations: {', '.join(patterns['dangerous_operations'])}")

        # Ordering patterns
        if patterns['ordering_patterns'].get('has_call_before_state_change'):
            query_parts.append("external call before state modification")

        # Keywords
        if patterns['keywords']:
            query_parts.append(f"keywords: {', '.join(patterns['keywords'][:5])}")

        return " | ".join(query_parts)


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

    extractor = StructuralPatternExtractor()
    patterns = extractor.extract_patterns(test_code)

    print("Structural Patterns Extracted:")
    print("=" * 80)
    print(f"External Calls: {len(patterns['external_calls'])}")
    for call in patterns['external_calls']:
        print(f"  Line {call['line_number']}: {call['call_type']}")

    print(f"\nState Changes: {len(patterns['state_changes'])}")
    for change in patterns['state_changes']:
        print(f"  Line {change['line_number']}: {change['variable']} ({change['operation']})")

    print(f"\nOrdering Patterns:")
    for key, value in patterns['ordering_patterns'].items():
        print(f"  {key}: {value}")

    print(f"\nRetrieval Query:")
    print(f"  {extractor.format_for_retrieval(patterns)}")

    print("\n" + "=" * 80)
    print("NOTE: No vulnerability classification!")
    print("This is just STRUCTURAL DESCRIPTION of the code.")
    print("RAG + LLM will discover what these patterns mean by comparing")
    print("to similar vulnerable code examples in the database.")
