# Setwise Ranking Paradigm — Implementation

> 論文實作：*A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models* (SIGIR 2024)

## 專案結構

```
report/
├── main.py                    # 主程式：執行所有排序方法並比較
├── pointwise.py               # Pointwise 排序（獨立評分）
├── pairwise.py                # Pairwise 排序（Heapsort / BubbleSort / AllPair）
├── listwise.py                # Listwise 排序（Generation / Likelihood）
├── setwise.py                 # Setwise 排序（Direct / Heapsort / BubbleSort）⭐
├── evaluation.py              # 評估指標（NDCG@k, Precision@k, Recall@k, MRR）
├── requirements.txt           # Python 套件依賴
├── paper_reading_report.md    # 論文閱讀報告
└── README.md                  # 本文件
```

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 執行比較實驗

```bash
# 使用預設模型 (flan-t5-base) 執行所有方法
python main.py

# 使用較大模型
python main.py --model google/flan-t5-large

# 自訂參數
python main.py --top_k 5 --c 4 --model google/flan-t5-base

# 僅執行特定方法
python main.py --methods setwise_direct setwise_heap pointwise
```

## 實作的排序方法

### 1. Pointwise (`pointwise.py`)
- **原理**：對每個文件獨立評分，提取 Yes/No logit 機率
- **複雜度**：O(N) forward passes
- **特點**：速度快、支援 batching，但缺乏跨文件比較

### 2. Pairwise (`pairwise.py`)
- **原理**：兩兩文件比較
- **支援排序策略**：
  - AllPair：O(N²-N)，exhaustive
  - Heapsort：O(k·log₂N)，early stopping
  - Bubble Sort：O(k·N)

### 3. Listwise (`listwise.py`)
- **Generation**：LLM 生成完整排序序列（sliding window）
- **Likelihood**：透過 logit extraction 直接排序（Setwise 增強版）

### 4. Setwise (`setwise.py`) ⭐
- **原理**：將 c 個候選文件作為無序集合輸入 LLM，透過 logit 提取獲得 multi-way preference
- **支援排序策略**：
  - **Direct**：單次前向傳播（適用於 N ≤ 26）
  - **Heapsort**：c-ary heap，O(k·log_c(N))
  - **Bubble Sort**：滑動窗口 c，O(k·N/(c-1))

## 核心概念

### Setwise Prompt 結構

```
Given a query '{query}', which of the following passages is more relevant?
Passage A: {passage_1}
Passage B: {passage_2}
Passage C: {passage_3}
...
Output only the passage label of the most relevant passage:
```

模型不需要生成答案——只需提取 A/B/C 標籤的 **logit 分數**，即可在**一次前向傳播**中完成多文件排序。

### 為什麼 Setwise 更快？

| 方法 | Forward Passes | 使用 Logits | 支援 Batching | Early Stopping |
|------|---------------|------------|--------------|----------------|
| Pointwise | O(N) | ✅ | ✅ | ❌ |
| Pairwise Heapsort | O(k·log₂N) | ❌ | ❌ | ✅ |
| Listwise Generation | O(r·N/s) | ❌ | 部分 | ✅ |
| **Setwise Heapsort** | **O(k·log_c(N))** | **✅** | **✅** | **✅** |

## 評估指標

- **NDCG@k**：Normalized Discounted Cumulative Gain
- **Precision@k**：前 k 個結果中相關文件的比例
- **Recall@k**：前 k 個結果涵蓋的相關文件比例
- **MRR**：Mean Reciprocal Rank

## 參考論文

Zhuang, S., Zhuang, H., Koopman, B., & Zuccon, G. (2024). *A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.* SIGIR '24.
