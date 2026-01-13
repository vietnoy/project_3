"""
RAG Retriever - Heavy Retrieval for Vulnerability Discovery
===========================================================

The CORE of the RAG-heavy philosophy: intelligent, multi-strategy retrieval
from dual vector databases to find similar vulnerable code patterns.

Retrieval weight: 70% of system intelligence
Feature extraction: 30%

Strategies:
1. Code similarity search (GraphCodeBERT)
2. Structural pattern matching (BGE text search)
3. Keyword fallback
4. Hybrid merging and deduplication
"""
from typing import List, Dict, Optional
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class RAGRetriever:
    """
    Heavy retrieval using dual vector databases.

    This is where the RAG magic happens - finding similar vulnerable
    code patterns from the database.
    """

    def __init__(
        self,
        code_db_path: str = './databases/code_db',
        text_db_path: str = './databases/text_db',
        min_similarity: float = 0.65
    ):
        """
        Initialize dual vector databases

        Args:
            code_db_path: Path to GraphCodeBERT code database
            text_db_path: Path to BGE-Large text database
            min_similarity: Minimum similarity threshold for filtering results
        """
        print("Initializing RAG Retriever...")
        print(f"  Code DB: {code_db_path}")
        print(f"  Text DB: {text_db_path}")
        print(f"  Similarity threshold: {min_similarity}")

        self.min_similarity = min_similarity

        # Load code database (GraphCodeBERT)
        print("  Loading GraphCodeBERT embeddings...")
        self.code_embeddings = HuggingFaceEmbeddings(
            model_name="microsoft/graphcodebert-base"
        )
        self.code_db = Chroma(
            persist_directory=code_db_path,
            embedding_function=self.code_embeddings
        )

        # Load text database (BGE-Large)
        print("  Loading BGE-Large embeddings...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )
        self.text_db = Chroma(
            persist_directory=text_db_path,
            embedding_function=self.text_embeddings
        )

        print("RAG Retriever ready")
        print()

    def retrieve(
        self,
        code: str,
        structural_patterns: Dict,
        k_code: int = 8,
        k_text: int = 8,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Heavy retrieval using multiple strategies

        Args:
            code: The user's code to analyze
            structural_patterns: Extracted structural patterns from code
            k_code: Number of similar code snippets to retrieve
            k_text: Number of similar text descriptions to retrieve
            verbose: Print retrieval progress

        Returns:
            List of retrieved findings with metadata
        """
        if verbose:
            print("RAG Retrieval")
            print("=" * 80)

        all_findings = []
        seen_ids = set()

        # Strategy 1: Code Similarity Search
        if verbose:
            print("Strategy 1: Code similarity search (GraphCodeBERT)...")

        code_results = self._retrieve_similar_code(code, k_code, verbose)

        for doc in code_results:
            finding_id = doc.metadata.get('finding_id')
            if finding_id and finding_id not in seen_ids:
                all_findings.append(self._format_finding(doc, 'code_similarity'))
                seen_ids.add(finding_id)

        if verbose:
            print(f"  Retrieved {len(code_results)} code snippets")
            print()

        # Strategy 2: Structural Pattern Matching
        if verbose:
            print("Strategy 2: Structural pattern matching (BGE text search)...")

        pattern_query = self._build_pattern_query(structural_patterns)
        if verbose:
            print(f"  Query: {pattern_query[:100]}...")

        text_results = self._retrieve_by_patterns(pattern_query, k_text, verbose)

        for doc in text_results:
            finding_id = doc.metadata.get('finding_id')
            if finding_id and finding_id not in seen_ids:
                all_findings.append(self._format_finding(doc, 'pattern_match'))
                seen_ids.add(finding_id)

        if verbose:
            print(f"  Retrieved {len(text_results)} pattern matches")
            print()

        # Strategy 3: Keyword Fallback (if needed)
        if len(all_findings) < 5:
            if verbose:
                print("Strategy 3: Keyword fallback (insufficient results)...")

            keywords = structural_patterns.get('keywords', [])
            keyword_query = ' '.join(keywords[:10])

            if keyword_query:
                keyword_results = self._retrieve_by_keywords(keyword_query, k_text, verbose)

                for doc in keyword_results:
                    finding_id = doc.metadata.get('finding_id')
                    if finding_id and finding_id not in seen_ids:
                        all_findings.append(self._format_finding(doc, 'keyword_match'))
                        seen_ids.add(finding_id)

                if verbose:
                    print(f"  Retrieved {len(keyword_results)} keyword matches")
                    print()

        # Summary
        if verbose:
            print("=" * 80)
            print(f"Total unique findings retrieved: {len(all_findings)}")
            print("=" * 80)
            print()

        return all_findings

    def _retrieve_similar_code(self, code: str, k: int, verbose: bool) -> List:
        """Strategy 1: Find code with similar structure using GraphCodeBERT"""
        try:
            # Get results with similarity scores
            results_with_scores = self.code_db.similarity_search_with_score(
                code,
                k=k*2  # Retrieve more, then filter
            )

            # Filter by similarity threshold (lower distance = higher similarity)
            # ChromaDB returns distance, not similarity (lower is better)
            # Typical range: 0.0-2.0, where 0 is identical
            # min_similarity 0.65 means we want distance < 0.7
            max_distance = 1.0 - self.min_similarity
            filtered_with_scores = [(doc, score) for doc, score in results_with_scores if score < max_distance][:k]

            if verbose and len(filtered_with_scores) < len(results_with_scores[:k]):
                print(f"  Filtered {len(results_with_scores[:k]) - len(filtered_with_scores)} low-similarity results")

            # Attach similarity scores to metadata
            for doc, score in filtered_with_scores:
                doc.metadata['similarity_score'] = round(1.0 - score, 3)  # Convert distance to similarity

            return [doc for doc, _ in filtered_with_scores]
        except Exception as e:
            if verbose:
                print(f"  Warning: Code search failed: {e}")
            return []

    def _retrieve_by_patterns(self, query: str, k: int, verbose: bool) -> List:
        """Strategy 2: Find findings matching structural patterns"""
        try:
            results_with_scores = self.text_db.similarity_search_with_score(
                query,
                k=k*2  # Retrieve more, then filter
            )

            # Filter by similarity threshold
            max_distance = 1.0 - self.min_similarity
            filtered_with_scores = [(doc, score) for doc, score in results_with_scores if score < max_distance][:k]

            if verbose and len(filtered_with_scores) < len(results_with_scores[:k]):
                print(f"  Filtered {len(results_with_scores[:k]) - len(filtered_with_scores)} low-similarity results")

            # Attach similarity scores to metadata
            for doc, score in filtered_with_scores:
                doc.metadata['similarity_score'] = round(1.0 - score, 3)  # Convert distance to similarity

            filtered = [doc for doc, _ in filtered_with_scores]

            return filtered
        except Exception as e:
            if verbose:
                print(f"  Warning: Pattern search failed: {e}")
            return []

    def _retrieve_by_keywords(self, query: str, k: int, verbose: bool) -> List:
        """Strategy 3: Keyword-based fallback retrieval"""
        try:
            results_with_scores = self.text_db.similarity_search_with_score(
                query,
                k=k*2  # Retrieve more, then filter
            )

            # Filter by similarity threshold
            max_distance = 1.0 - self.min_similarity
            filtered = [doc for doc, score in results_with_scores if score < max_distance][:k]

            return filtered
        except Exception as e:
            if verbose:
                print(f"  Warning: Keyword search failed: {e}")
            return []

    def _build_pattern_query(self, structural_patterns: Dict) -> str:
        """
        Convert structural patterns into a text query for retrieval.

        Example:
        Input: {external_calls: [...], state_changes: [...]}
        Output: "external call before state modification, public function..."
        """
        query_parts = []

        # External calls
        if structural_patterns.get('external_calls'):
            calls = structural_patterns['external_calls']
            call_types = list(set([c['call_type'] for c in calls]))
            query_parts.append(f"code with {', '.join(call_types)}")

        # Ordering patterns (very important for reentrancy, etc.)
        ordering = structural_patterns.get('ordering_patterns', {})
        if ordering.get('has_call_before_state_change'):
            query_parts.append("external call before state update")
        if ordering.get('has_state_change_before_call'):
            query_parts.append("state update before external call")

        # Dangerous operations
        if structural_patterns.get('dangerous_operations'):
            ops = structural_patterns['dangerous_operations']
            query_parts.append(' '.join(ops))

        # Functions
        if structural_patterns.get('functions'):
            functions = structural_patterns['functions']
            visibility = list(set([f['visibility'] for f in functions]))
            query_parts.append(f"{', '.join(visibility)} function")

            # Check for lack of modifiers (important pattern)
            functions_without_modifiers = [f for f in functions if not f['modifiers']]
            if functions_without_modifiers:
                query_parts.append("function without access control")

        # Control structures
        if structural_patterns.get('control_structures'):
            structs = structural_patterns['control_structures']
            struct_types = list(set([s['type'] for s in structs]))
            query_parts.append(f"contains {', '.join(struct_types)}")

        # Keywords
        if structural_patterns.get('keywords'):
            keywords = structural_patterns['keywords'][:5]
            query_parts.append(' '.join(keywords))

        # Build final query
        query = ' '.join(query_parts)

        # If query is empty, use a generic fallback
        if not query:
            query = "smart contract vulnerability code pattern"

        return query

    def _format_finding(self, doc, retrieval_strategy: str) -> Dict:
        """Format a retrieved document into a finding dict"""
        metadata = doc.metadata
        content = doc.page_content

        return {
            'finding_id': metadata.get('finding_id', 'unknown'),
            'title': metadata.get('title', 'N/A'),
            'content': content[:1000],  # First 1000 chars
            'full_content': content,
            'firm_name': metadata.get('firm_name', 'Unknown'),
            'protocol_name': metadata.get('protocol_name', 'Unknown'),
            'impact': metadata.get('impact', 'MEDIUM'),
            'reference_link': metadata.get('reference_link', ''),
            'doc_type': metadata.get('doc_type', 'unknown'),
            'retrieval_strategy': retrieval_strategy,
            'source': metadata.get('source', '')
        }

    def get_finding_by_id(self, finding_id: str) -> Optional[Dict]:
        """
        Retrieve full finding by ID from text database.

        Useful for enriching code-based retrievals with full context.
        """
        try:
            # Search text database for this finding_id
            results = self.text_db.similarity_search(
                "",  # Empty query
                k=1,
                filter={'finding_id': finding_id}
            )

            if results:
                return self._format_finding(results[0], 'id_lookup')

            return None

        except Exception as e:
            print(f"Warning: Could not retrieve finding {finding_id}: {e}")
            return None


# Example usage
if __name__ == '__main__':
    from structural_extractor import StructuralPatternExtractor

    test_code = """
    function withdraw() public {
        uint256 amount = balances[msg.sender];
        (bool success,) = msg.sender.call{value: amount}("");
        require(success);
        balances[msg.sender] = 0;
    }
    """

    # Extract structural patterns
    extractor = StructuralPatternExtractor()
    patterns = extractor.extract_patterns(test_code)

    print("Extracted Structural Patterns:")
    print("=" * 80)
    print(f"External calls: {len(patterns['external_calls'])}")
    print(f"State changes: {len(patterns['state_changes'])}")
    print(f"Ordering: Call before state change = {patterns['ordering_patterns'].get('has_call_before_state_change')}")
    print()

    # Retrieve similar findings
    print("Retrieving Similar Vulnerable Code...")
    print("=" * 80)
    print()

    retriever = RAGRetriever()
    findings = retriever.retrieve(test_code, patterns, k_code=5, k_text=5)

    print()
    print("Retrieved Findings:")
    print("=" * 80)
    for i, finding in enumerate(findings, 1):
        print(f"\n[Finding {i}] {finding['title']}")
        print(f"  ID: {finding['finding_id']}")
        print(f"  Firm: {finding['firm_name']}")
        print(f"  Impact: {finding['impact']}")
        print(f"  Strategy: {finding['retrieval_strategy']}")
        print(f"  Content: {finding['content'][:200]}...")

    print()
    print("=" * 80)
    print("These findings will be given to the LLM to DISCOVER")
    print("what vulnerability pattern the user's code matches.")
