"""
Pointwise Ranking with LLM
Each document is independently scored against the query.
Uses logit extraction on Yes/No tokens for relevance scoring.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple


POINTWISE_PROMPT = (
    "Passage: {passage}\n"
    "Query: {query}\n"
    "Does the passage answer the query? Answer 'Yes' or 'No':"
)


class PointwiseRanker:
    """
    Pointwise: Score each document independently by extracting
    the logit probability of 'Yes' vs 'No' tokens.
    Forward passes required: O(N)
    """

    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Pre-compute token IDs for 'Yes' and 'No'
        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

    def _score_single(self, query: str, passage: str) -> float:
        """Score a single query-passage pair using logit extraction."""
        prompt = POINTWISE_PROMPT.format(query=query, passage=passage)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # Use decoder_input_ids for the first generated token
        decoder_input_ids = self.tokenizer("<pad>", return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[0, -1, :]  # logits for the first generated token

        # Extract Yes/No probabilities
        yes_no_logits = logits[[self.yes_token_id, self.no_token_id]]
        probs = torch.softmax(yes_no_logits, dim=0)
        return probs[0].item()  # P(Yes)

    def rank(self, query: str, passages: List[str]) -> List[Tuple[int, float]]:
        """
        Rank passages by relevance to query.
        Returns list of (original_index, score) sorted by score descending.
        """
        scores = []
        for i, passage in enumerate(passages):
            score = self._score_single(query, passage)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def rank_batch(self, query: str, passages: List[str], batch_size: int = 8) -> List[Tuple[int, float]]:
        """Batch scoring for better GPU utilization."""
        prompts = [POINTWISE_PROMPT.format(query=query, passage=p) for p in passages]

        all_scores = []
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            inputs = self.tokenizer(
                batch_prompts, return_tensors="pt", truncation=True,
                max_length=512, padding=True
            ).to(self.device)

            decoder_input_ids = self.tokenizer(
                ["<pad>"] * len(batch_prompts),
                return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, -1, :]

            for j in range(logits.size(0)):
                yes_no_logits = logits[j][[self.yes_token_id, self.no_token_id]]
                probs = torch.softmax(yes_no_logits, dim=0)
                all_scores.append((batch_start + j, probs[0].item()))

        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores
