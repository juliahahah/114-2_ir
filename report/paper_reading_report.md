# Paper Reading Report

> **課程：** 114-2 資訊檢索 (Information Retrieval)  
> **報告人：** 劉怡妏
> **日期：** 2026/03/20  

---

## 一、論文基本資訊

| 欄位 | 內容 |
|------|------|
| **論文標題** | A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models |
| **作者** | Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, Guido Zuccon |
| **發表場域** | SIGIR 2024 |
| **關鍵字** | Zero-shot Ranking, Large Language Models, Document Re-ranking, Prompt Engineering, Logit Extraction |

---

## 二、研究動機與問題定義

### 2.1 背景

近年來，大型語言模型（LLM）被廣泛應用於零樣本（zero-shot）文件排序任務。現有方法主要分為三種 prompting 範式：

| 範式 | 作法 | 優點 | 缺點 |
|------|------|------|------|
| **Pointwise** | 對每個文件獨立評分（Yes/No logits） | 速度快、支援 batching、可用 logit extraction | 缺乏跨文件比較能力，排序品質較低 |
| **Pairwise** | 兩兩文件比較，選出較相關者 | 排序精確度最高 | 複雜度 $O(N^2 - N)$，計算成本極高，無法規模化 |
| **Listwise** | 將多文件一起輸入，生成完整排序序列 | 具備多文件上下文感知 | 依賴 autoregressive token generation，延遲高、有格式錯誤風險 |

### 2.2 核心問題

若以排序品質（NDCG@10）為 Y 軸、計算效率（延遲/成本）為 X 軸，現有三種方法分佈於不同象限，但**右上角「高品質 + 高效率」的象限始終無人佔領**——作者稱之為 **"The Missing Quadrant"**。

本論文的核心研究問題為：

> **能否設計一種新的 prompting 範式，同時達到 Pairwise 等級的排序品質與 Pointwise 等級的計算效率？**

---

## 三、方法論 (Methodology)

### 3.1 Setwise Prompting：核心設計

Setwise 的核心思想是將一組 $c$ 個候選文件作為**無序集合**（而非有序列表）呈現給 LLM，並要求模型僅輸出最相關文件的標籤。其 prompt 結構如下：

```
Given a query {query}, which of the following passages is more relevant to the query?
Passage A: {passage_1}
Passage B: {passage_2}
Passage C: {passage_3}
...
Output only the passage label of the most relevant passage:
```

關鍵創新在於：**不需要模型實際生成答案**。透過提取模型在各標籤 token（A、B、C…）上的 logit 分數，即可在單次前向傳播中獲得所有候選文件的相對偏好排序（multi-way preference）。

### 3.2 三大範式的融合

Setwise 同時繼承了三種既有範式的優勢：

1. **來自 Pairwise 的高效排序與 Early Stopping**：可搭配 heapsort / bubble sort 等排序演算法，僅需找到 top-k 即提前終止。
2. **來自 Listwise 的多文件上下文感知**：一次輸入 $c$ 個文件，讓模型在比較時具備全局視野。
3. **來自 Pointwise 的 Logit Extraction**：單次前向傳播直接取得排序信號，完全跳過耗時的 token generation。

### 3.3 演算法層面的優化

#### 3.3.1 Setwise Heapify

傳統 Pairwise Heapify 使用二元堆（binary heap），每個節點僅與 2 個子節點比較。Setwise 將比較集合擴展至 $c$ 個節點：

- **Pairwise Heapify**（9 個節點）：至少需要 **6 次比較**
- **Setwise Heapify**（$c=4$，9 個節點）：僅需 **2 次比較**

透過增加每次比較的 fan-out，排序樹的深度被大幅壓扁，LLM 推論次數顯著降低。

#### 3.3.2 Setwise Bubble Sort

傳統 Pairwise Bubble Sort 每次比較相鄰 2 個元素，複雜度為 $O(k \cdot N)$。Setwise Bubble Sort 使用大小為 $c$ 的滑動窗口：

$$O(k \cdot N) \longrightarrow O\left(k \cdot \frac{N}{c-1}\right)$$

當 $c=4$ 時，每輪迭代的步數降為原來的 $\frac{1}{3}$。

#### 3.3.3 Listwise Likelihood Estimation

Setwise 也可重新詮釋 Listwise prompt：不做 token generation，而是直接提取每個文件標籤的 logit likelihood。

| 方法 | 運作方式 | 特性 |
|------|---------|------|
| **Listwise Generation（傳統）** | LLM 逐 token 生成排序序列 | 高延遲、有格式錯誤風險 |
| **Listwise Likelihood via Setwise** | 單次前向傳播提取各標籤 logit | 零延遲增量、零格式錯誤 |

---

## 四、實驗設計與結果

### 4.1 實驗設定

- **評測資料集**：TREC DL 2019、TREC DL 2020
- **評估指標**：NDCG@10
- **對比方法**：Pointwise、Pairwise（AllPair / Heapsort）、Listwise Generation
- **測試模型**：Flan-T5、LLaMA-2、Vicuna、GPT-3.5（Commercial API）

### 4.2 主要實驗結果

#### 4.2.1 效率 vs. 品質

**Setwise.heapsort 相比 Pairwise 達成：**

- 計算成本減少 **62%**
- NDCG@10 差異僅 **0.8%**（可忽略）

在 Latency vs. NDCG@10 散點圖中，Setwise 配置穩定佔據**左上角最佳區域**（低延遲、高品質），而 Pairwise 延遲普遍達 30–60 秒以上。

#### 4.2.2 對初始排序的魯棒性（Robustness）

實驗測試了三種初始排序條件：

| 初始排序 | Listwise Generation NDCG@10 | Setwise NDCG@10 |
|---------|------------------------------|-----------------|
| Baseline BM25 | ~0.80 | ~0.80 |
| Inverted BM25 | ~0.10 | ~0.80 |
| Random BM25 | ~0.10 | ~0.80 |

Listwise Generation 對初始排序極度敏感，品質崩跌至 0.1；而 Setwise 因將文件視為**無序集合**，表現幾乎不受影響，穩定維持在 0.8 左右。

#### 4.2.3 跨模型通用性與經濟可擴展性

| 模型 | Setwise 表現 |
|------|-------------|
| **Flan-T5** | 查詢延遲壓至 ~8 秒，全面超越 Pairwise / Listwise |
| **LLaMA-2 & Vicuna** | 維持最優 NDCG@10，避開自迴歸生成瓶頸 |
| **GPT-3.5 API** | Listwise 每查詢成本 \$0.045 → Setwise **\$0.029**（降低約 36%） |

---

## 五、論文架構限制分析

| 面向 | 限制描述 |
|------|---------|
| **候選集合大小 $c$** | $c$ 值受 LLM context window 限制，文件過長時可容納的候選數量有限 |
| **Logit 可取得性** | 商業 API（如 GPT-3.5）不一定開放 logit 存取，需改用 generation-based 替代方案 |
| **文件截斷** | 長文件需截斷以適應 context window，可能損失關鍵資訊 |
| **Zero-shot 限定** | 未探討 few-shot 或 fine-tuned 設定下的效果 |

---

## 六、個人評論與反思

### 6.1 優點

1. **問題定位精準**：清楚指出現有三種範式都無法同時兼顧品質與效率，並以 "Missing Quadrant" 的視覺化方式呈現，非常具有說服力。

2. **設計優雅**：Setwise 的核心思想——將「生成答案」轉化為「表達偏好」——是一個看似簡單但極為深刻的 insight。僅透過改變 prompt 結構與推論方式，就能從根本上改變系統的效率特性。

3. **實驗全面**：涵蓋多個資料集、多種模型架構、多種排序演算法、以及魯棒性分析，結果一致且具有說服力。

4. **實用價值高**：在 GPT-3.5 API 上的成本實驗直接展示了 real-world 的經濟效益，對工業界有直接參考價值。

### 6.2 可改進方向

1. **更大規模模型的驗證**：論文使用的最大模型為 GPT-3.5，未測試 GPT-4 或更新的模型。隨著模型能力提升，Setwise 的優勢是否仍然顯著值得探討。

2. **候選集合大小的自適應策略**：$c$ 值目前為固定超參數，未來可探索根據查詢難度或文件長度動態調整 $c$ 的策略。

3. **與其他檢索階段的整合**：論文聚焦於 re-ranking 階段，若能探討 Setwise 與 dense retrieval 或 learned sparse retrieval 的端到端整合，將更具系統性。

4. **多語言場景**：實驗僅在英文資料集上進行，跨語言排序的適用性仍待驗證。

### 6.3 對資訊檢索領域的啟示

這篇論文最重要的啟示在於：**prompt 的結構設計本身就是一種架構決策**。在 LLM-based IR 系統中，選擇讓模型「生成排序」還是「表達偏好」，會從根本上決定系統的效率上限。Setwise 證明了透過巧妙的 prompt 設計與 logit 利用，可以打破傳統範式間的 trade-off。

---

## 七、結論

本論文提出的 Setwise prompting 範式，透過三項關鍵創新——多文件集合式 prompt、logit extraction 取代 token generation、以及高效排序演算法的適配——成功填補了 LLM 零樣本排序中「高品質 + 高效率」的空白象限。實驗結果在 TREC DL 2019/2020 上驗證了其有效性：相比 Pairwise 方法減少 62% 計算成本、NDCG@10 僅差 0.8%；相比 Listwise 在 GPT-3.5 上節省約 36% API 成本。此外，Setwise 對初始排序的魯棒性使其在實際部署中具有顯著優勢。

---

## 參考文獻

- Zhuang, S., Zhuang, H., Koopman, B., & Zuccon, G. (2024). A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models. *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)*.

---
