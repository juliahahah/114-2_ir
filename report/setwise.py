"""
Setwise Ranking with LLM — The Core Innovation

Feeds an unordered set of c candidate documents to the LLM,
extracts multi-way preference via logit scores on label tokens (A, B, C, ...),
achieving cross-document comparison in a single forward pass.

Supports three sorting strategies:
  - Setwise Heapsort (c-ary heap)
  - Setwise Bubble Sort (sliding window of size c)
  - Setwise direct (single-pass logit ranking for small sets)
"""

import string
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple, Dict


SETWISE_PROMPT = (
    "Given a query '{query}', which of the following passages is more relevant to the query?\n"
    "{passages_text}\n"
    "Output only the passage label of the most relevant passage:"
)

# Labels: A, B, C, D, ...
LABELS = list(string.ascii_uppercase)


def _build_setwise_passages(passages: List[str], indices: List[int]) -> str:
    """Format passages with labels Passage A, Passage B, ..."""
    lines = []
    for i, idx in enumerate(indices):
        lines.append(f"Passage {LABELS[i]}: {passages[idx]}")
    return "\n".join(lines)


class SetwiseRanker:
    """
    Setwise Paradigm: The hybrid engine combining:
      - Pairwise's efficient sorting & early stopping
      - Listwise's multi-document contextual awareness
      - Pointwise's lightning-fast logit extraction

    Key: Expand comparison set from 2 (pairwise) to c documents,
    drastically flattening the sorting tree depth.
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Pre-compute token IDs for labels A-Z
        self.label_token_ids = {}
        for label in LABELS:
            self.label_token_ids[label] = self.tokenizer.encode(label, add_special_tokens=False)[0]

        self.inference_count = 0

    def _get_preference(self, query: str, passages: List[str], indices: List[int]) -> List[Tuple[int, float]]:
        """
        Core Setwise operation: Given c candidate passages,
        extract multi-way preference via logit scores in a single forward pass.

        Returns: list of (original_index, logit_score) sorted by score descending.
        """
        c = len(indices)
        passages_text = _build_setwise_passages(passages, indices)
        prompt = SETWISE_PROMPT.format(query=query, passages_text=passages_text)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        decoder_input_ids = self.tokenizer("<pad>", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1, :]

        self.inference_count += 1

        # Extract logits for each label token
        scored = []
        for i in range(c):
            label = LABELS[i]
            token_id = self.label_token_ids[label]
            scored.append((indices[i], logits[token_id].item()))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _get_winner(self, query: str, passages: List[str], indices: List[int]) -> int:
        """Get the index of the most relevant passage from a set."""
        scored = self._get_preference(query, passages, indices)
        return scored[0][0]

    # ──────────────────────────────────────────────
    #  Strategy 1: Direct (single-pass for small N)
    # ──────────────────────────────────────────────

    def rank_direct(self, query: str, passages: List[str]) -> List[Tuple[int, float]]:
        """
        Direct ranking: single forward pass for up to 26 passages.
        Returns: list of (index, score) sorted by score descending.
        """
        self.inference_count = 0
        n = min(len(passages), 26)  # max 26 labels (A-Z)
        indices = list(range(n))
        return self._get_preference(query, passages, indices)

    # ──────────────────────────────────────────────
    #  Strategy 2: Setwise Heapsort (c-ary heap)
    # ──────────────────────────────────────────────

    def rank_heapsort(self, query: str, passages: List[str], c: int = 4, top_k: int = 10) -> List[int]:
        """
        Setwise Heapsort: c-ary max-heap.
        Each sift-down compares c children simultaneously instead of 2.

        Pairwise heapify (9 nodes): 6 comparisons
        Setwise heapify (c=4, 9 nodes): 2 comparisons

        Forward passes: O(k * log_c(N))
        """
        self.inference_count = 0
        n = len(passages)
        heap = list(range(n))

        def _sift_down(arr, root, end):
            """Sift down using c-ary comparison."""
            while True:
                first_child = c * root + 1
                if first_child > end:
                    break

                # Gather root + all children for setwise comparison
                children = []
                for i in range(c):
                    child_idx = first_child + i
                    if child_idx <= end:
                        children.append(child_idx)

                if not children:
                    break

                # Compare root against all children simultaneously
                candidates = [root] + children
                candidate_passage_indices = [arr[ci] for ci in candidates]

                # Use setwise to find the winner
                winner_passage_idx = self._get_winner(query, passages, candidate_passage_indices)

                # Find which position in candidates won
                winner_pos = None
                for ci in candidates:
                    if arr[ci] == winner_passage_idx:
                        winner_pos = ci
                        break

                if winner_pos == root:
                    break

                arr[root], arr[winner_pos] = arr[winner_pos], arr[root]
                root = winner_pos

        # Build heap (heapify)
        last_parent = (n - 2) // c
        for i in range(last_parent, -1, -1):
            _sift_down(heap, i, n - 1)

        # Extract top-k
        result = []
        end = n - 1
        for _ in range(min(top_k, n)):
            result.append(heap[0])
            heap[0], heap[end] = heap[end], heap[0]
            end -= 1
            if end > 0:
                _sift_down(heap, 0, end)

        return result

    # ──────────────────────────────────────────────
    #  Strategy 3: Setwise Bubble Sort
    # ──────────────────────────────────────────────

    def rank_bubblesort(self, query: str, passages: List[str], c: int = 4, top_k: int = 10) -> List[int]:
        """
        Setwise Bubble Sort: sliding window of size c.

        Pairwise Bubble Sort: O(k * N)
        Setwise Bubble Sort:  O(k * N/(c-1))

        Each window comparison covers c elements, advancing by (c-1) steps.
        """
        self.inference_count = 0
        n = len(passages)
        indices = list(range(n))
        step = c - 1  # Each window advances by c-1

        for k in range(min(top_k, n)):
            # Bubble the best element to position k
            pos = n - 1
            while pos > k:
                window_end = pos
                window_start = max(k, pos - c + 1)
                window_size = window_end - window_start + 1

                if window_size < 2:
                    pos -= 1
                    continue

                # Get passage indices for current window
                window_indices = [indices[j] for j in range(window_start, window_end + 1)]

                # Find winner
                winner_passage_idx = self._get_winner(query, passages, window_indices)

                # Move winner to the front (leftmost position) of the window
                for j in range(window_start, window_end + 1):
                    if indices[j] == winner_passage_idx:
                        # Swap to window_start
                        indices[window_start], indices[j] = indices[j], indices[window_start]
                        break

                pos -= step

        return indices[:top_k]
