# PEFA 論文口頭報告講稿

> 課程:114-2 Information Retrieval
> 論文:PEFA — Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models (WSDM '24)
> 對應簡報:`PEFA_report.pptx` (共 15 張投影片)
> 預估時間:每頁約 50 秒 ~ 1 分鐘,整體約 12–15 分鐘

---

## 投影片 1 — 封面

各位老師、同學大家好,我是 314581027 劉怡妏。

今天要報告的論文是 WSDM 2024 的一篇 Amazon 研究團隊的文章,標題是
**PEFA: Parameter-Free Adapters for Large-scale Embedding-based Retrieval Models**。

中文可以翻譯成「無參數適配器,用於大規模 embedding 檢索模型」。

這篇論文的核心問題,是要解決 dense retrieval 領域中
「fine-tuning 太貴、又不適用於 black-box LLM」的痛點。
作者群來自 Amazon 與 UCLA,第一作者 Wei-Cheng Chang 也是 PECOS 框架的主要貢獻者,
程式碼開源在 amzn/pecos 上。

接下來十幾分鐘,我會帶大家走過這個方法的動機、設計、實驗與我的個人觀察。

---

## 投影片 2 — Outline 大綱

簡報共分八個段落:

1. 第一段先講 **Motivation**,為什麼 full-parameter fine-tuning 在工業界其實非常痛苦。
2. 第二段是 **Preliminaries**,簡單回顧 dense retrieval、bi-encoder、MIPS 與 ANN。
3. 第三段是論文的核心想法,**PEFA 框架** — 一個 ERM 與 kNN 的凸組合。
4. 第四段細看兩個變體 **PEFA-XL 與 PEFA-XS**,它們的 trade-off 截然不同。
5. 第五段是 **時空複雜度分析**。
6. 第六、第七段是兩組實驗:**文件檢索 (NQ-320K / Trivia-QA)** 與 **產品搜尋 (ProdSearch)**。
7. 最後一段是 **Ablation 與相關研究的討論**。

我會盡量在每一段都帶出一個關鍵 take-away,讓大家不只是看到結果,而是看到設計理由。

---

## 投影片 3 — Motivation

先講動機。

目前 SoTA 的 dense retrieval 通常採用 bi-encoder,
但要把預訓練模型 fine-tune 到接近 SoTA,實際 pipeline 非常複雜:
要做 pre-training、用 random / BM25 negatives 做第一階段 fine-tune,
再用 hard negatives 做第二階段、最後還要做 cross-encoder distillation。
在 Amazon 這種電商規模,query-product pair 動輒幾十億,
全參數 fine-tune 隨便就要上千 GPU 小時。

更重要的是,所有這些方法都要存取「模型梯度」 —
這對 GPT-3、Claude 這種 black-box LLM 是不可能的。
而且傳統 ERM 在 long-tail query 上也常常失準。

PEFA 的承諾就放在右邊:**不做反向傳播、不需要梯度、CPU 幾小時建 ANN index 就好**,
而且不論是 pre-trained 或 fine-tuned ERM 都能用。

這是這篇論文最吸引我的地方:它把 adaptation 重新框架成「ANN 工程問題」,
而不是「優化問題」。

---

## 投影片 4 — Preliminaries

進入正式內容之前,快速複習一下 dense retrieval 的基本設定。

ERM,也就是 bi-encoder,把 query 和 passage 各自編碼成一個 d 維向量,
relevance 用內積或 cosine similarity 表示。在這篇論文裡,
作者假設所有 embedding 都做了 ℓ2 normalization,所以內積等同於 cosine。

學習面看左邊:訓練資料就是 relevant (q, p) pairs,
最大化 softmax log-likelihood。但是分母是整個 corpus,
所以實務上必須做 negative sampling — 隨機、BM25、hard mining,
每一階段都會引入新的訓練成本。

推論面看右邊:給定一個 query embedding,要從 corpus 裡找最相關的 k 個 passage,
這是 Maximum Inner Product Search,簡稱 MIPS。Naive 計算是 O(n),
n 是 corpus size,十億級無法接受。
所以實務都用 HNSW、FAISS、ScaNN 這類 ANN library,
把推論複雜度壓到 O(log n)。

關鍵觀察是:**建 ANN index 跑在 CPU 上,成本遠低於 GPU fine-tuning**。
這個對比就是 PEFA 之所以「便宜」的根基。

---

## 投影片 5 — PEFA 框架核心

這張投影片是整篇論文的核心公式。

PEFA 的 scoring function 就是兩項加權:

$$f_{\text{PEFA}}(\hat{q}, p_j) = \lambda \cdot f_{\text{ERM}} + (1 - \lambda) \cdot f_{\text{kNN}}$$

其中 λ 介於 0 到 1 之間,用來平衡兩個 scoring function。
重點是:**這個融合只在 inference time 發生**,
模型參數 θ 從頭到尾都沒被動到。

下面三張卡片總結三個重要性質:

第一,**Parameter-free** — 沒有任何優化步驟,「學習」就只是建一個存 key-value pair 的 ANN index。

第二,**Black-box friendly** — 只需要 forward embedding,
所以可以套用在像 GPT-3 那種無法存取梯度的 LLM encoder 上。

第三,**Orthogonal** — 跟 pre-training、fine-tuning、甚至 PEFT (例如 LoRA、Adapter) 都不衝突,可以疊加使用。

那麼這個 kNN 怎麼建?有兩種選擇,就是 PEFA-XL 和 PEFA-XS。

---

## 投影片 6 — PEFA-XL

PEFA-XL 是比較直接的版本。

它的想法是:給定一個 test query q-hat,
在「訓練 query 空間 Q」裡找最相似的 top-k 個 training query,
然後把這些鄰居對應的相關 passage 聚合起來。
形式上,對角矩陣 D_{i,i} = 1 if i ∈ NN(q-hat, Q; k),否則為 0。

代回去就得到 Equation 6,公式上半部是原本的 ERM 內積,
下半部是 kNN 的鄰居加權和。
要注意的是,因為 f_ERM 是 cosine 在 [-1, 1] 之間,
所以 kNN 那項要除以 k 做歸一化,讓兩邊量級可比。

看右邊架構圖,這是論文的 Figure 1。
PEFA-XL 需要**兩個 ANN index**:
一個建在 passage 空間 P,size = n;
一個建在訓練 query 空間 Q,size = m;
然後用 ⟨ q-hat, passage ⟩ 取 ERM 分數,
用 ⟨ q-hat, q_i ⟩ 經由 relevance matrix Y 取得 kNN 分數,
最後線性加權。

這也就是它叫 "XL" 的原因:兩個 index 意味著 storage 與 latency 都會放大。

---

## 投影片 7 — PEFA-XS

PEFA-XS 是更聰明的版本。

關鍵 trick:把 NN(q-hat, Q; k) 換成 I(p_j, Y),
也就是「passage p_j 的所有 relevant queries」。
注意!這個集合**只跟 passage 有關,跟 test query 完全無關**。

這代表什麼?既然 kNN 那項與 q-hat 無關,
我們可以把 λ p_j + (1−λ) Π(Q^T Y_{:,j}) 預先算好,
這就是 Equation 8 的整個內積形式。
Π 是 ℓ2 normalization,把向量推到單位球面上。

換句話說:**整個融合可以在 offline 建 index 時就完成**,
線上只剩下一次 ANN search,inference 成本跟原 ERM 完全一樣。

右邊是論文 Figure 2,只剩一個 ANN index,
存的就是「interpolated passage embedding」。

這個設計非常漂亮 — XS 用一個簡單的「neighborhood 重新定義」
就把 XL 的兩個 index 折成一個,部署上幾乎零代價。
這個 trick 也跟 XMC 領域的 PIFA embeddings 概念相通。

---

## 投影片 8 — 複雜度分析

把三個方法的成本攤開對照:

| 方法     | Build time         | Index size                              | Inference         |
|----------|--------------------|-----------------------------------------|-------------------|
| ERM      | O(n log n)         | O(nd + |E_P|)                           | O(log n)          |
| PEFA-XS  | O(n log n)         | O(nd + |E_P|)                           | O(log n)          |
| PEFA-XL  | O(n log n + m log m) | O(nd + |E_P| + md + |E_Q| + nnz(Y)) | O(log n + log m)  |

第一個 take-away:**PEFA-XS 跟 baseline ERM 完全同等成本**,
所以可以無痛 drop-in。

第二:**PEFA-XL 多了第二個 HNSW**,在電商資料裡 query 數 m 常常比 product 數 n 還大,
所以這個 overhead 不可忽略。

第三:儲存 relevance matrix Y 只需要 O(nnz(Y)),因為 Y 是 sparse 的。

第四,也是最關鍵的:**整個 index build 都跑在 CPU**,
跟 GPU fine-tuning 比起來便宜好幾個數量級。

---

## 投影片 9 — NQ-320K 結果

接下來看實驗。

NQ-320K 是 Natural Questions 的版本,32 萬個 query-document pair。
這個實驗的設定是:**ERM 已經做過 full fine-tune**,
PEFA 再疊加上去 — 也就是說我們是要證明
「就算 ERM 已經很強,PEFA 還能加多少分」。

看表格中間,MPNet_base 本身 R@10 是 80.82、R@100 是 92.39,
加上 PEFA-XL 之後跳到 88.72 / 95.13,
**直接超越前 SoTA NCI 的 85.20 / 92.42**,
而且 NCI 是 Seq2Seq decoder 結構,訓練成本高很多。

底部兩列是平均增益:PEFA-XS 平均 +9.22 R@10,PEFA-XL +11.82。
GTR_base 也是類似情況,R@10 從 79.74 拉到 88.71。

右邊我列了三點 headline finding,並提醒大家一個重要觀察:
**Seq2Seq indexer 如 NCI、DSI、SEAL 需要昂貴的 decoder 訓練,
而 PEFA 只是建一個 HNSW index 就能達到 SoTA。**

---

## 投影片 10 — Trivia-QA 與 Ablation

左邊是 Trivia-QA 的結果。
這個實驗的設定**不一樣**:ERM 沒有經過 fine-tune,只是 pre-trained。
也就是說,我們在挑戰 PEFA 的「泛化能力」。

結果驚人:Sent-BERT_distill 原本 R@20 只有 51.94,
加上 PEFA-XS 直接跳到 86.28,**漲了 34 個百分點**。
GTR_base 也從 71.75 漲到 85.30。
平均 PEFA-XS 提升 +18.67、PEFA-XL +17.07,
這顯示 PEFA 不只在 fine-tuned 模型上有效,
**對 pre-trained-only 的 black-box ERM 增益更大**。

右邊是超參數 ablation。
首先 λ 是內插係數:對 PEFA-XS,最佳值大約在 0.3 到 0.5;
對 PEFA-XL,最佳值在 0.1 附近 — 也就是 kNN 那項的權重更高。
其次 k 是鄰居數,**k = 32 通常就飽和了**,
再大反而會引入雜訊。

最後當 λ = 1 時,PEFA 退化回原 ERM,作為 sanity check。

---

## 投影片 11 — Product Search Recall

接著是工業界規模的實驗:Amazon 內部的 product search。

三個資料集:5M、15M、30M 對應 corpus 大小,
最大那個有 5 億個 query-product pair。
為了保護隱私,**表格裡的數字都是相對於 MPNet baseline 的 Recall 絕對增益**,
而不是絕對值。

最值得看的是最下面三列:**FT-ERM 是 Amazon 內部 carefully fine-tune 過的模型**,
代表生產環境的 SoTA。即便在這麼強的 baseline 上,
PEFA-XS 在 30M 還能加 +5.3% R@100,**PEFA-XL 加 +14.5%**,
還有 +5.9% / +5.9% 的 R@1000 增益。

看上面幾組:把 PEFA-XL 疊加到 pre-trained 的 GTR 或 E5,
30M 數據集上 R@100 增益達 31.71 / 31.91,
**已經超越完整 fine-tune 的 FT-ERM**。
這個結果非常有 punch — 等於說「不用 fine-tune,
只要 PEFA 就能做出比 fine-tune 還好的模型」。

---

## 投影片 12 — Product Search 部署成本

但 recall 不是唯一的衡量,
工業界部署最關心的是 disk size、build time 與 latency。

以最大的 ProdSearch-30M 為例:
- FT-ERM 本身 51.9 GiB,build 0.9 小時,latency 0.77 ms/query。
- **PEFA-XS 完全一樣的 51.9 GiB**,build 也是 1 小時,latency 甚至略低 (0.71 ms)。
- **PEFA-XL 漲到 287.7 GiB(3.6 倍)**,build 4.7 小時,latency 1.99 ms (約 2.4 倍)。

右邊兩個總結:
PEFA-XS 是**「免費的午餐」** — 拿到 R@100 +5.3 的同時,index 與 latency 完全不變。
PEFA-XL 是**「付費升級」** — 漲 3.6 倍 index、2.4 倍 latency,換來更大的 R@100 +14.5。

這裡作者很實際:**讓 practitioner 自己決定 trade-off**。
對 latency 敏感的線上服務選 XS,
對精準度要求極高、能負擔額外硬體成本的場景選 XL。

順帶一提,4.7 小時的 CPU build 比起 FT-ERM 的「上百 GPU 小時」
還是便宜了 1-2 個數量級。

---

## 投影片 13 — Effect of Supervised Data

這是新加入的一張,對應論文 Section 5.4 的 Figure 3。

橫軸是「sampled 多少比例的 relevance matrix Y」,從 5% 到 95%。
上排兩張圖是 ΔRecall@100 vs FT-ERM,下排是 index 大小。

兩個關鍵 finding:

第一,**PEFA-XS (藍線) 只用 10% 的訓練 pair 就能追平 FT-ERM**,
而且 index 大小幾乎不增加 — 這對資料稀缺場景很重要。

第二,**PEFA-XL (紅線) 在 5% 的取樣比例下就有顯著增益**,
但代價是 index 比 FT-ERM 大約 1.4 倍,而且隨著資料增加會線性成長。

右下角的 callout 是 deployment 角度的總結:
PEFA-XL 的 latency 在所有 dataset size 下都穩定維持在 FT-ERM 的 2 倍左右,
不會因為資料量增加而爆炸。

這張圖的意義在於:**PEFA 對 supervised data 的需求其實很低**,
意味著它在資料蒐集成本高的領域 (例如醫療、法律檢索) 也有實用性。

---

## 投影片 14 — Related Work

把 PEFA 放在文獻地圖上看四個方向。

**第一,kNN-LM**(Khandelwal 等):
在 language model 的 vocabulary 上做 kNN,概念上跟 PEFA-XL 類似 —
都是「parametric 模型 + 非參數 kNN 內插」。
差別在 kNN-LM 的 vocab 很小,可以做 exact lookup;
而 PEFA-XL 的 passage 空間 P 與 query 空間 Q 都很大,
所以必須用 ANN 處理。

**第二,PIFA (XMC 領域)**:
Positive Instance Feature Aggregation,
把訓練 query 聚合成 passage 的另一種表示,
傳統用在 sparse tfidf clustering。
**PEFA-XS 是這個概念的 dense embedding 版本**,
而且還做了線性內插並做 ANN search,設計更完整。

**第三,Parameter-efficient fine-tuning** (Adapter / LoRA / Prefix-tuning):
這些方法在 retrieval 上效果不顯著 (Tam et al.),
而且仍然需要梯度。**PEFA 完全不需要梯度**,所以可以用在 black-box LLM,
而且跟 PEFT 是正交的,可以疊加。

**第四,倫理面**:
作者特別提到 PEFA 的 kNN 元件帶來**可解釋性** —
可以追溯是哪些 training query 影響了檢索結果。
隱私方面,則使用「年度聚合的匿名 log」,個別 session 無法被回溯。

---

## 投影片 15 — 結論與我的看法

最後總結。

左邊是論文的結論:
- PEFA 是一個 parameter-free、gradient-free 的 ERM adapter 框架。
- PEFA-XS 零部署成本、中等增益;PEFA-XL 增益較大但 latency 與 index 約 2 倍。
- 適用於 pre-trained 與 fine-tuned ERM 雙場景。
- NQ-320K 達成新 SoTA;Amazon 30M 產品檢索上對 FT-ERM 再加 5–14%。

右邊是我自己的觀點:

**Strength 一:重新框架問題**。
把「fine-tune 一個模型」變成「建一個更聰明的 ANN index」,
這個視角轉換有意義 — 因為 ANN 工具鏈 (HNSW, FAISS) 在工業界已經非常成熟。

**Strength 二:可解釋性**。
傳統 ERM 是黑盒子,而 PEFA 的 kNN 元件可以告訴你
「哪些 training query 貢獻了這個檢索結果」,這在 debug 與合規上都很有用。

**Limitation 一:PEFA-XL 需要 label-side Y 在推論時可用**,
所以不是真正的 zero-shot 方法。

**Limitation 二:當 Y 非常稀疏時,儲存 Q 與 Y 的成本可能會 dominate**,
這在 cold-start 場景需要小心。

**Open question**:
- 跟 re-ranking 階段的組合?
- query side 的 augmentation 能不能也用類似想法?
- Y 隨時間更新時,如何 incremental update HNSW index?

以上就是我的報告,謝謝聆聽,有問題歡迎提出。

---

## 附錄 · 可能會被問到的問題與答覆

**Q1: PEFA 跟 RAG (Retrieval-Augmented Generation) 是不是同一回事?**
不是。RAG 是「先檢索,再讓 LLM 生成」;PEFA 是「在檢索階段本身用 kNN 強化 ranking」。
不過兩者可以結合 — PEFA 改善的 retrieval recall,自然能讓 RAG 的下游生成更準。

**Q2: 為什麼 PEFA-XL 在 R@10 增益比 R@100 大?**
直覺上,kNN 強化的是「最相似那一小撮 candidate 的精準排序」,
所以在 small-k 的 Recall 增益更明顯。
R@100 已經接近天花板,提升空間自然較小。

**Q3: 如果訓練 query 本身有偏差或品質不均,PEFA 會不會放大?**
會。這是 kNN-based 方法的共通弱點 — garbage in, garbage out。
論文在 ethical considerations 也暗示了這點,
所以實務上 Y 的清洗仍然重要。

**Q4: λ 要怎麼挑?有沒有 adaptive 方法?**
論文用 grid search 找最佳 λ;
對 PEFA-XS 大約 0.3-0.5,PEFA-XL 約 0.1。
是否能 per-query 自適應 λ 是 open question,
作者沒有探討,我覺得是一個值得後續研究的方向。
