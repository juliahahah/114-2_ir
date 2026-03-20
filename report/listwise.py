"""
Listwise Ranking with LLM
Two variants:
  1. Generation-based: LLM generates a full ranking sequence (traditional)
  2. Likelihood-based: Use logit extraction on document labels (Setwise-enhanced)
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple


LISTWISE_PROMPT = (
    "Given a query '{query}', rank the following passages from most relevant to least relevant.\n"
    "{passages_text}\n"
    "Rank the passages by outputting their identifiers from most to least relevant. "
    "Output only the identifiers separated by ' > ':"
)


def _build_passages_text(passages: List[str], offset: int = 0) -> str:
    """Format passages with labels [1], [2], ..."""
    lines = []
    for i, p in enumerate(passages):
        lines.append(f"[{offset + i + 1}] {p}")
    return "\n".join(lines)


class ListwiseGenerationRanker:
    """
    Listwise Generation: LLM autoregressively generates a full ranking sequence.
    Uses sliding window approach.
    Forward passes: O(r * (N/s)) where r = rounds, s = window size.
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.inference_count = 0

    def _rank_window(self, query: str, passages: List[str], indices: List[int]) -> List[int]:
        """Rank a window of passages using generation."""
        passages_text = ""
        for i, idx in enumerate(indices):
            passages_text += f"[{i + 1}] {passages[idx]}\n"

        prompt = LISTWISE_PROMPT.format(query=query, passages_text=passages_text.strip())
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        self.inference_count += 1

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse the generated ranking: expect format like "[2] > [1] > [3]"
        numbers = re.findall(r'\d+', decoded)

        ranked_indices = []
        seen = set()
        for num_str in numbers:
            num = int(num_str) - 1  # 0-indexed within window
            if 0 <= num < len(indices) and num not in seen:
                ranked_indices.append(indices[num])
                seen.add(num)

        # Append any missing indices at the end
        for i, idx in enumerate(indices):
            if idx not in ranked_indices:
                ranked_indices.append(idx)

        return ranked_indices

    def rank(self, query: str, passages: List[str], window_size: int = 10, step: int = 5, rounds: int = 3) -> List[int]:
        """
        Sliding window ranking.
        Returns indices in ranked order.
        """
        self.inference_count = 0
        n = len(passages)
        indices = list(range(n))

        for _ in range(rounds):
            # Slide from bottom to top
            end = n
            while end > 0:
                start = max(0, end - window_size)
                window_indices = indices[start:end]
                if len(window_indices) > 1:
                    reranked = self._rank_window(query, passages, window_indices)
                    indices[start:end] = reranked
                end -= step

        return indices


class ListwiseLikelihoodRanker:
    """
    Listwise Likelihood via Setwise: Instead of generating tokens,
    extract logit probabilities for each document label.
    One forward pass. Zero formatting errors.
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def rank(self, query: str, passages: List[str]) -> List[Tuple[int, float]]:
        """
        Rank passages by extracting logit likelihoods for each label token.
        Returns list of (original_index, logit_score) sorted by score descending.
        """
        passages_text = _build_passages_text(passages)
        prompt = LISTWISE_PROMPT.format(query=query, passages_text=passages_text)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        decoder_input_ids = self.tokenizer("<pad>", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1, :]

        # Extract logits for label tokens: "1", "2", "3", ...
        scores = []
        for i in range(len(passages)):
            label = str(i + 1)
            token_id = self.tokenizer.encode(label, add_special_tokens=False)[0]
            scores.append((i, logits[token_id].item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
