"""
Main runner: Compare all four ranking paradigms on a sample dataset.

Usage:
    python main.py                          # Run with built-in demo data
    python main.py --model google/flan-t5-base  # Use a smaller model
    python main.py --top_k 5 --c 3          # Customize parameters
"""

import argparse
import time
import json
from typing import List, Dict

from evaluation import evaluate_ranking, ndcg_at_k


# ─────────────────────────────────────────────────────
#  Sample Data (for demonstration without external datasets)
# ─────────────────────────────────────────────────────

SAMPLE_QUERY = "What are the health benefits of green tea?"

SAMPLE_PASSAGES = [
    # 0 - Highly relevant (grade 3)
    "Green tea is rich in antioxidants called catechins, which can help reduce cell damage. "
    "Studies have shown that regular consumption of green tea may lower the risk of heart disease, "
    "improve brain function, and aid in weight loss.",

    # 1 - Relevant (grade 2)
    "Tea has been consumed for thousands of years. Green tea, in particular, contains polyphenols "
    "that have anti-inflammatory properties and may help protect against certain types of cancer.",

    # 2 - Somewhat relevant (grade 1)
    "Caffeine is found in many beverages including coffee, tea, and energy drinks. Green tea "
    "contains less caffeine than coffee but enough to produce a mild stimulant effect.",

    # 3 - Not relevant (grade 0)
    "The stock market experienced significant volatility last quarter, with technology stocks "
    "leading the declines amid rising interest rates and inflation concerns.",

    # 4 - Relevant (grade 2)
    "Research published in the Journal of Nutrition found that green tea extract can boost "
    "metabolic rate and increase fat burning. The EGCG compound in green tea is particularly "
    "effective at promoting thermogenesis.",

    # 5 - Not relevant (grade 0)
    "Python is a popular programming language known for its simplicity and readability. "
    "It is widely used in data science, web development, and artificial intelligence.",

    # 6 - Somewhat relevant (grade 1)
    "Herbal teas such as chamomile and peppermint are caffeine-free alternatives to traditional "
    "teas. While green tea has health benefits, herbal teas offer different therapeutic properties.",

    # 7 - Highly relevant (grade 3)
    "A meta-analysis of 13 randomized controlled trials found that green tea significantly "
    "reduces blood pressure, total cholesterol, and LDL cholesterol levels. These cardiovascular "
    "benefits are attributed to the high concentration of epigallocatechin gallate (EGCG).",

    # 8 - Not relevant (grade 0)
    "The annual conference on machine learning will be held in Vancouver next month. "
    "Researchers from around the world will present papers on deep learning and NLP.",

    # 9 - Somewhat relevant (grade 1)
    "Drinking too much green tea can cause side effects due to its caffeine content, "
    "including anxiety, insomnia, and digestive issues. Moderation is recommended.",
]

# Ground truth relevance grades
RELEVANCE = {0: 3, 1: 2, 2: 1, 3: 0, 4: 2, 5: 0, 6: 1, 7: 3, 8: 0, 9: 1}


def run_pointwise(model_name: str, query: str, passages: List[str]) -> dict:
    """Run Pointwise ranking."""
    from pointwise import PointwiseRanker

    print("\n" + "=" * 60)
    print("  POINTWISE RANKING")
    print("  O(N) forward passes | Logit extraction | Supports batching")
    print("=" * 60)

    ranker = PointwiseRanker(model_name=model_name)

    start = time.time()
    results = ranker.rank(query, passages)
    elapsed = time.time() - start

    ranked_indices = [idx for idx, _ in results]
    print(f"\n  Ranked order: {ranked_indices}")
    print(f"  Scores: {[(idx, f'{score:.4f}') for idx, score in results]}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: {len(passages)}")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=10)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": "Pointwise", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": len(passages), "metrics": metrics}


def run_pairwise(model_name: str, query: str, passages: List[str], top_k: int) -> dict:
    """Run Pairwise Heapsort ranking."""
    from pairwise import PairwiseRanker

    print("\n" + "=" * 60)
    print("  PAIRWISE RANKING (Heapsort)")
    print("  O(k * log2(N)) forward passes | Token generation")
    print("=" * 60)

    ranker = PairwiseRanker(model_name=model_name)

    start = time.time()
    ranked_indices = ranker.rank_heapsort(query, passages, top_k=top_k)
    elapsed = time.time() - start

    print(f"\n  Top-{top_k} ranked: {ranked_indices}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: {ranker.inference_count}")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=top_k)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": "Pairwise (Heapsort)", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": ranker.inference_count, "metrics": metrics}


def run_listwise_generation(model_name: str, query: str, passages: List[str]) -> dict:
    """Run Listwise Generation ranking."""
    from listwise import ListwiseGenerationRanker

    print("\n" + "=" * 60)
    print("  LISTWISE GENERATION RANKING")
    print("  O(r * N/s) forward passes | Token generation | Sliding window")
    print("=" * 60)

    ranker = ListwiseGenerationRanker(model_name=model_name)

    start = time.time()
    ranked_indices = ranker.rank(query, passages, window_size=5, step=3, rounds=2)
    elapsed = time.time() - start

    print(f"\n  Ranked order: {ranked_indices}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: {ranker.inference_count}")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=10)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": "Listwise (Generation)", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": ranker.inference_count, "metrics": metrics}


def run_listwise_likelihood(model_name: str, query: str, passages: List[str]) -> dict:
    """Run Listwise Likelihood ranking (Setwise-enhanced)."""
    from listwise import ListwiseLikelihoodRanker

    print("\n" + "=" * 60)
    print("  LISTWISE LIKELIHOOD RANKING (via Setwise)")
    print("  1 forward pass | Logit extraction | Zero formatting errors")
    print("=" * 60)

    ranker = ListwiseLikelihoodRanker(model_name=model_name)

    start = time.time()
    results = ranker.rank(query, passages)
    elapsed = time.time() - start

    ranked_indices = [idx for idx, _ in results]
    print(f"\n  Ranked order: {ranked_indices}")
    print(f"  Scores: {[(idx, f'{score:.4f}') for idx, score in results]}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: 1")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=10)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": "Listwise (Likelihood)", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": 1, "metrics": metrics}


def run_setwise_heapsort(model_name: str, query: str, passages: List[str], c: int, top_k: int) -> dict:
    """Run Setwise Heapsort ranking."""
    from setwise import SetwiseRanker

    print("\n" + "=" * 60)
    print(f"  SETWISE RANKING (Heapsort, c={c})")
    print(f"  O(k * log_{c}(N)) forward passes | Logit extraction")
    print("=" * 60)

    ranker = SetwiseRanker(model_name=model_name)

    start = time.time()
    ranked_indices = ranker.rank_heapsort(query, passages, c=c, top_k=top_k)
    elapsed = time.time() - start

    print(f"\n  Top-{top_k} ranked: {ranked_indices}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: {ranker.inference_count}")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=top_k)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": f"Setwise (Heapsort, c={c})", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": ranker.inference_count, "metrics": metrics}


def run_setwise_bubblesort(model_name: str, query: str, passages: List[str], c: int, top_k: int) -> dict:
    """Run Setwise Bubble Sort ranking."""
    from setwise import SetwiseRanker

    print("\n" + "=" * 60)
    print(f"  SETWISE RANKING (Bubble Sort, c={c})")
    print(f"  O(k * N/(c-1)) forward passes | Logit extraction")
    print("=" * 60)

    ranker = SetwiseRanker(model_name=model_name)

    start = time.time()
    ranked_indices = ranker.rank_bubblesort(query, passages, c=c, top_k=top_k)
    elapsed = time.time() - start

    print(f"\n  Top-{top_k} ranked: {ranked_indices}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: {ranker.inference_count}")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=top_k)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": f"Setwise (BubbleSort, c={c})", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": ranker.inference_count, "metrics": metrics}


def run_setwise_direct(model_name: str, query: str, passages: List[str]) -> dict:
    """Run Setwise Direct ranking (single pass)."""
    from setwise import SetwiseRanker

    print("\n" + "=" * 60)
    print("  SETWISE RANKING (Direct — single forward pass)")
    print("  1 forward pass | Full multi-way preference | Logit extraction")
    print("=" * 60)

    ranker = SetwiseRanker(model_name=model_name)

    start = time.time()
    results = ranker.rank_direct(query, passages)
    elapsed = time.time() - start

    ranked_indices = [idx for idx, _ in results]
    print(f"\n  Ranked order: {ranked_indices}")
    print(f"  Scores: {[(idx, f'{score:.4f}') for idx, score in results]}")
    print(f"  Latency: {elapsed:.2f}s")
    print(f"  Forward passes: 1")

    metrics = evaluate_ranking(ranked_indices, RELEVANCE, k=10)
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    return {"method": "Setwise (Direct)", "ranked": ranked_indices, "latency": elapsed,
            "forward_passes": 1, "metrics": metrics}


def print_summary(all_results: List[dict]):
    """Print a comparison summary table."""
    print("\n")
    print("=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n  {'Method':<35} {'NDCG@10':>10} {'Latency':>10} {'FW Passes':>12}")
    print("  " + "-" * 70)
    for r in all_results:
        ndcg = r["metrics"].get("NDCG@10", r["metrics"].get("NDCG@5", 0))
        print(f"  {r['method']:<35} {ndcg:>10.4f} {r['latency']:>9.2f}s {r['forward_passes']:>12}")
    print()

    # Ideal ranking
    ideal = sorted(RELEVANCE.keys(), key=lambda x: RELEVANCE[x], reverse=True)
    print(f"  Ideal ranking (by relevance): {ideal}")
    print(f"  Relevance grades: {RELEVANCE}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Setwise Ranking Paradigm - Comparison Demo")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="HuggingFace model name (default: google/flan-t5-base)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top documents to retrieve (default: 5)")
    parser.add_argument("--c", type=int, default=4,
                        help="Setwise comparison set size (default: 4)")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["pointwise", "pairwise", "listwise_gen", "listwise_ll",
                                 "setwise_direct", "setwise_heap", "setwise_bubble"],
                        help="Methods to run")
    args = parser.parse_args()

    print("=" * 80)
    print("  Setwise Ranking Paradigm — Implementation & Comparison")
    print("  Paper: 'A Setwise Approach for Effective and Highly Efficient")
    print("          Zero-shot Ranking with Large Language Models'")
    print(f"  Model: {args.model}")
    print(f"  Passages: {len(SAMPLE_PASSAGES)} | Top-k: {args.top_k} | c: {args.c}")
    print("=" * 80)
    print(f"\n  Query: \"{SAMPLE_QUERY}\"")

    all_results = []

    method_runners = {
        "pointwise": lambda: run_pointwise(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES),
        "pairwise": lambda: run_pairwise(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES, args.top_k),
        "listwise_gen": lambda: run_listwise_generation(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES),
        "listwise_ll": lambda: run_listwise_likelihood(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES),
        "setwise_direct": lambda: run_setwise_direct(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES),
        "setwise_heap": lambda: run_setwise_heapsort(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES, args.c, args.top_k),
        "setwise_bubble": lambda: run_setwise_bubblesort(args.model, SAMPLE_QUERY, SAMPLE_PASSAGES, args.c, args.top_k),
    }

    for method in args.methods:
        if method in method_runners:
            try:
                result = method_runners[method]()
                all_results.append(result)
            except Exception as e:
                print(f"\n  [ERROR] {method}: {e}")

    if all_results:
        print_summary(all_results)

        # Save results to JSON
        output = {
            "query": SAMPLE_QUERY,
            "model": args.model,
            "num_passages": len(SAMPLE_PASSAGES),
            "results": []
        }
        for r in all_results:
            output["results"].append({
                "method": r["method"],
                "ranked_indices": r["ranked"],
                "latency_seconds": round(r["latency"], 3),
                "forward_passes": r["forward_passes"],
                "metrics": {k: round(v, 4) for k, v in r["metrics"].items()}
            })

        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print("  Results saved to results.json")


if __name__ == "__main__":
    main()
