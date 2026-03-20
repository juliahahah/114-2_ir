"""
Pairwise Ranking with LLM
Compares documents head-to-head in pairs.
Supports AllPair (exhaustive) and Heapsort-based ranking.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple


PAIRWISE_PROMPT = (
    "Given a query '{query}', which of the following two passages is more relevant to the query?\n"
    "Passage A: {passage_a}\n"
    "Passage B: {passage_b}\n"
    "Output only 'A' or 'B':"
)


class PairwiseRanker:
    """
    Pairwise: Compare documents head-to-head in pairs.
    AllPair forward passes: O(N^2 - N)
    Heapsort forward passes: O(k * log2(N))
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.token_a_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.token_b_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.inference_count = 0

    def _compare(self, query: str, passage_a: str, passage_b: str) -> str:
        """Compare two passages, return 'A' or 'B' (the more relevant one)."""
        prompt = PAIRWISE_PROMPT.format(query=query, passage_a=passage_a, passage_b=passage_b)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        decoder_input_ids = self.tokenizer("<pad>", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1, :]

        self.inference_count += 1
        ab_logits = logits[[self.token_a_id, self.token_b_id]]
        return "A" if ab_logits[0] > ab_logits[1] else "B"

    def rank_allpair(self, query: str, passages: List[str]) -> List[Tuple[int, int]]:
        """
        AllPair: Exhaustive pairwise comparison with win counting.
        O(N^2 - N) forward passes.
        Returns list of (original_index, win_count) sorted by wins descending.
        """
        self.inference_count = 0
        n = len(passages)
        wins = [0] * n

        for i in range(n):
            for j in range(i + 1, n):
                result = self._compare(query, passages[i], passages[j])
                if result == "A":
                    wins[i] += 1
                else:
                    wins[j] += 1

        ranked = [(i, wins[i]) for i in range(n)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def rank_heapsort(self, query: str, passages: List[str], top_k: int = 10) -> List[int]:
        """
        Heapsort-based ranking with early stopping.
        O(k * log2(N)) forward passes.
        Returns top-k indices in ranked order.
        """
        self.inference_count = 0
        n = len(passages)
        indices = list(range(n))

        # Build a max-heap (heapify)
        def _sift_down(arr, start, end):
            root = start
            while True:
                left = 2 * root + 1
                if left > end:
                    break
                # Find the larger child
                swap = root
                if self._compare(query, passages[arr[swap]], passages[arr[left]]) == "B":
                    swap = left
                right = left + 1
                if right <= end:
                    if self._compare(query, passages[arr[swap]], passages[arr[right]]) == "B":
                        swap = right
                if swap == root:
                    break
                indices[root], indices[swap] = indices[swap], indices[root]
                root = swap

        # Build heap
        for start in range((n - 2) // 2, -1, -1):
            _sift_down(indices, start, n - 1)

        # Extract top-k
        result = []
        end = n - 1
        for _ in range(min(top_k, n)):
            result.append(indices[0])
            indices[0], indices[end] = indices[end], indices[0]
            end -= 1
            if end > 0:
                _sift_down(indices, 0, end)

        return result

    def rank_bubblesort(self, query: str, passages: List[str], top_k: int = 10) -> List[int]:
        """
        Bubble sort-based ranking.
        O(k * N) forward passes.
        Returns top-k indices in ranked order.
        """
        self.inference_count = 0
        n = len(passages)
        indices = list(range(n))

        for k in range(min(top_k, n)):
            for j in range(n - 1, k, -1):
                result = self._compare(query, passages[indices[j - 1]], passages[indices[j]])
                if result == "B":
                    indices[j - 1], indices[j] = indices[j], indices[j - 1]

        return indices[:top_k]
