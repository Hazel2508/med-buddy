# MedBuddy RAG 知识库 — 完整设计思路

> 此文档记录项目从数据获取到对话接入的完整技术方案，包含每步的原理、设计决策和实现细节。
> 与 `TROUBLESHOOTING.md` 配套使用：本文档讲「应该怎么做和为什么」，TROUBLESHOOTING 讲「遇到了什么坑和怎么绕开」。

---

## 一、项目背景与目标

### 问题
MedBuddy 是一个面向美国慢性病患者的用药管理 App，内置 AI 问诊对话。  
AI 直接回答用药问题存在两个风险：
1. **幻觉（Hallucination）**：模型可能捏造用法、剂量或禁忌
2. **时效性**：模型训练数据有截止日期，说明书版本可能已更新

### 解决方案：RAG（Retrieval-Augmented Generation）
在用户提问和 LLM 生成之间，插入一个「检索」步骤：
1. 先从 FDA 官方说明书中找到最相关的文本片段
2. 把片段作为 context 塞进 prompt，让 LLM「基于原文回答」

这样既保留了 LLM 的自然语言能力，又把事实性答案锚定在可信的官方文档上。

### 目标药品（15 种慢性病常用药）
| 药品 | 类别 | 剂型 |
|------|------|------|
| metformin | 2 型糖尿病（双胍类） | tablet ER |
| glipizide | 2 型糖尿病（磺脲类） | tablet |
| dapagliflozin | 2 型糖尿病（SGLT-2 抑制剂） | tablet |
| sitagliptin | 2 型糖尿病（DPP-4 抑制剂） | tablet |
| amlodipine | 高血压（钙通道阻滞剂） | tablet |
| lisinopril | 高血压（ACEI） | tablet |
| losartan | 高血压（ARB） | tablet |
| hydrochlorothiazide | 高血压（利尿剂） | tablet |
| atorvastatin | 高脂血症（他汀类） | tablet |
| rosuvastatin | 高脂血症（他汀类） | tablet |
| metoprolol | 高血压 / 心脏病（β 受体阻滞剂） | tablet |
| aspirin | 心血管预防 | tablet |
| levothyroxine | 甲状腺功能减退 | tablet |
| albuterol | 哮喘（支气管扩张剂） | aerosol |
| alendronate | 骨质疏松（双膦酸盐） | tablet |

---

## 二、系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    离线构建（Offline）                    │
│                                                         │
│  DailyMed API                                           │
│      │ Step 1: 下载 XML                                  │
│      ▼                                                  │
│  drug_labels/*.xml  (15 个 FDA 说明书)                   │
│      │ Step 2: 解析 + Chunking                           │
│      ▼                                                  │
│  chunks/all_chunks.jsonl  (772 个文本片段)               │
│      │ Step 3: Embedding + 存入向量库                    │
│      ▼                                                  │
│  faiss_index/  (向量库 + metadata)                       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ 在线查询（Online）
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Step 4: 对话接入                       │
│                                                         │
│  用户提问                                                │
│      │ embed 问题                                        │
│      │ FAISS 近邻搜索                                    │
│      │ 取 Top-K chunks                                   │
│      ▼                                                  │
│  Claude API  ←  [system prompt + chunks + 用户问题]      │
│      │                                                  │
│      ▼                                                  │
│  回答（附来源章节）                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 三、Step 1 — DailyMed XML 下载

**脚本：** `download_labels.py`  
**输出：** `drug_labels/{drug_name}.xml`（15 个文件，共约 2.9 MB）

### 数据来源
[DailyMed](https://dailymed.nlm.nih.gov) 是美国 NIH 维护的 FDA 药品说明书数据库，提供 REST API v2。

### API 流程

```
1. 搜索
   GET /services/v2/spls.json
       ?drug_name={name}
       &labeltypes=HUMAN%20PRESCRIPTION%20DRUG%20LABEL
       &pagesize=100

   响应字段：setid, title, published_date

2. 筛选最佳候选
   ① 从 title 中确认剂型关键字（TABLET 或 AEROSOL）
   ② title 必须以药品名开头（排除复方药，如 "GLYBURIDE AND METFORMIN"）
   ③ 选 published_date 最新的一条，取其 setid

3. 下载 zip
   GET https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm?setId={setid}
   （注意：v2 API 的 /spls/{setid}.zip 已废弃，302 重定向到首页）

4. 解压
   zip 包含 XML + 图片，只保留 .xml，重命名为 {drug_name}.xml
```

### 关键设计决策
- **用 `requests` 库而非系统 `curl`**：macOS LibreSSL 与 DailyMed TLS 有间歇性相容性问题，conda 的 `requests`（OpenSSL）稳定
- **排除复方药**：`title.upper().startswith(drug_name.upper())` 过滤组合成分说明书
- **`published_date` 即「更新日期」**：v2 API 搜索结果无 `updated_date` 字段，`published_date` 即最后发布/更新日期

---

## 四、Step 2 — XML 解析与 Chunking

**脚本：** `parse_chunks.py`  
**输出：** `chunks/all_chunks.jsonl`（772 个 chunks）

### SPL XML 格式
DailyMed 使用 **HL7 SPL**（Structured Product Labeling）格式，命名空间 `urn:hl7-org:v3`。

```xml
<document xmlns="urn:hl7-org:v3">
  <component>
    <structuredBody>
      <component>
        <section>                              <!-- 顶层章节 -->
          <code code="34067-9" .../>           <!-- LOINC 代码（章节类型） -->
          <title>1 INDICATIONS AND USAGE</title>
          <text>                               <!-- 正文 -->
            <paragraph>...</paragraph>
            <list><item>...</item></list>
          </text>
          <section>                            <!-- 子章节，可无限嵌套 -->
            <title>1.1 Type 2 Diabetes</title>
            <text>...</text>
          </section>
        </section>
      </component>
    </structuredBody>
  </component>
</document>
```

**重要陷阱**：`<section>` 不是 `<document>` 的直接子节点，嵌套在 `component/structuredBody/component` 层下。遍历时必须穿透所有非 section 元素，否则返回 0 个结果。

### 章节识别：LOINC 代码
每个 `<section>` 有标准 LOINC 代码，用于识别章节类型（不依赖 title 文字）：

| LOINC | 章节 |
|-------|------|
| 34066-1 | Boxed Warning（黑框警告）|
| 34067-9 | Indications and Usage（适应症）|
| 34068-7 | Dosage and Administration（用法用量）|
| 34070-3 | Contraindications（禁忌）|
| 43685-7 | Warnings and Precautions（警告）|
| 34084-4 | Adverse Reactions（不良反应）|
| 34073-7 | Drug Interactions（药物相互作用）|
| 43684-0 | Use in Specific Populations（特殊人群）|
| 34088-5 | Overdosage（过量）|
| 42230-3 | Patient Information（患者信息）|
| 51945-4 | Principal Display Panel（跳过：包装图）|
| 48780-1 | SPL Highlights（跳过：重复摘要）|

### Chunking 策略

```
原则：语义完整 > 主题单一 > 大小适中

level 1: 以 subsection（子章节）为基本切片单位
         → 每个 chunk = 一个临床问题的完整回答

level 2: chunk > 2000 chars → 按句子边界拆分
         相邻 chunk 保留 200 chars 重叠（overlap）
         → 防止答案在 chunk 边界被截断

level 3: chunk < 40 chars → 丢弃
         → 去除页眉、占位符等无意义碎片
```

**为什么 2000 chars？**  
text-embedding-3-small token 上限 8191（≈32k chars），2000 chars ≈ 500 tokens，搜索精度和 token 利用率的平衡点。

**Overlap 的作用：**  
```
问题的答案若恰好跨越两个 chunk 的边界 →
  无 overlap：两个 chunk 各自语义不完整，搜索可能都命中不了
  有 overlap：边界内容在相邻两个 chunk 均保留，至少其中一个能命中
```

### 输出格式

```json
{
  "drug":        "metformin",
  "loinc_code":  "43685-7",
  "section":     "5 WARNINGS AND PRECAUTIONS > 5.1 Lactic Acidosis",
  "chunk_index": 0,
  "text":        "Lactic acidosis is a rare but serious complication...",
  "char_count":  1842
}
```

`section` 字段保留层级路径，既用于回答时标注来源，也支持后续按章节过滤搜索范围。

### 最终统计
- 总 chunks：**772 个**（15 种药品）
- 平均大小：**1100 chars**（≈275 tokens）
- 最大 chunk：4516 chars（Clinical Studies 长句，仍在 token 限制内）
- 跳过的章节：Principal Display Panel、SPL Highlights（重复）

---

## 五、Step 3 — Embedding + 向量数据库

**脚本：** `embed_store.py`  
**输出：** `faiss_index/index.faiss` + `faiss_index/metadata.jsonl`

### 原理：语义搜索 vs 关键词搜索

```
关键词搜索：
  问题："Can I drink alcohol with metformin?"
  搜索："alcohol" AND "metformin"
  → 可能漏掉说明书里写 "ethanol" 或 "alcoholic beverages" 的段落

语义搜索（Embedding）：
  把问题和每个 chunk 都转成高维向量
  向量空间中语义相近的文字距离也近
  → "alcohol" 和 "ethanol" 的向量很接近，不会漏掉
```

### Embedding 模型选择
- **OpenAI text-embedding-3-small**（主选）
  - 维度：1536
  - Token 上限：8191
  - 优点：效果好，API 简单
  - 费用：$0.02 / 1M tokens（772 chunks × ~275 tokens ≈ 0.2M tokens ≈ $0.004）
  
- **BGE-small-en-v1.5**（备选，完全离线）
  - 维度：384
  - 完全本地运行，零费用
  - 效果略低于 OpenAI

### FAISS 向量库

FAISS（Facebook AI Similarity Search）是轻量、高效的本地向量相似度搜索库。

```
存储结构：
  faiss_index/
  ├── index.faiss       ← 772 个向量（float32, 1536 维）
  └── metadata.jsonl    ← 772 个 chunk 的完整 metadata（drug/section/text...）

搜索流程：
  query_text
    │ embed（同一模型）
    ▼
  query_vector (1536 维)
    │ FAISS.search(query_vector, k=5)
    ▼
  top-5 最近邻向量的索引 → 通过索引查 metadata.jsonl → 取回 text
```

### 相似度计算
FAISS 默认用 **L2 距离（欧式距离）**，也支持 **余弦相似度**（需先对向量做 L2 normalize）。  
对文本 embedding，余弦相似度通常效果更好（不受向量长度影响）。

---

## 六、Step 4 — 接入对话

**脚本：** `query.py`  
**依赖：** `anthropic`、`sentence-transformers`、`faiss-cpu`、`numpy`

### 查询流程

```
用户问题
  │ embed_query()  — BGE 编码，加 query 前缀
  ▼
query_vector（384 维，L2 归一化）
  │ index.search(q_vec, k)  — FAISS cosine 搜索
  ▼
top-k 候选 chunk（可选：按 drug 元数据过滤）
  │ build_context()  — 拼装 [DRUG — Section] 块，上限 12000 chars
  ▼
Claude API（claude-opus-4-8）
  system: 临床药剂师助手 prompt，强制"仅基于原文回答"
  ▼
回答（附来源章节引用 + 安全提醒）
```

### 核心实现

```python
def retrieve(question, model, index, metadata, top_k=5, drug_filter=None):
    prefixed = f"Represent this sentence: {question}"  # BGE query 侧前缀
    q_vec = model.encode([prefixed], normalize_embeddings=True).astype(np.float32)

    # drug_filter 开启时扩大搜索范围再过滤
    search_k = top_k * 6 if drug_filter else top_k
    scores, indices = index.search(q_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = metadata[idx]
        if drug_filter and chunk["drug"].lower() != drug_filter.lower():
            continue
        results.append({**chunk, "_score": float(score)})
        if len(results) >= top_k:
            break
    return results
```

### 关键 Prompt 设计原则
1. **"Answer ONLY based on..."**：防止模型用训练知识「补充」超出说明书的内容
2. **附来源标注**：回答中引用 `[DRUG — Section]`，让用户知道信息出处
3. **不确定时承认**：说明书未涵盖的问题，明确告知不在文档范围内
4. **安全提醒**：每条回答末尾提醒患者用药变更前咨询医生/药剂师

### CLI 使用方式
```bash
export ANTHROPIC_API_KEY="sk-ant-..."

python3 query.py                                       # 交互模式
python3 query.py -q "What are metformin side effects?" # 单次提问
python3 query.py --drug metformin -q "Can I drink?"    # 限定药品
python3 query.py --verbose                             # 显示 chunk 来源
```

### drug_filter 参数设计
用户在 App 中已选定药品 → 搜索时只在该药品的 chunks 中找 → 精度更高、不会混淆不同药品的信息。  
实现上先搜 `top_k * 6` 再过滤，保证即使前几名全是其他药品也能找到足够结果。

---

## 七、文件结构

```
medbuddy/
├── CLAUDE.md                  ← 项目背景（AI 读取的上下文）
├── PROJECT_DESIGN.md          ← 本文档：完整设计思路
├── TROUBLESHOOTING.md         ← 问题记录（持续更新）
│
├── download_labels.py         ← Step 1: DailyMed 下载脚本
├── parse_chunks.py            ← Step 2: XML 解析 + Chunking
├── embed_store.py             ← Step 3: Embedding + FAISS 构建
├── query.py                   ← Step 4: 查询接口（FAISS 检索 + Claude API）
│
├── drug_labels/               ← Step 1 输出（15 个 XML，2.9MB）
│   ├── metformin.xml
│   ├── atorvastatin.xml
│   └── ...
│
├── chunks/                    ← Step 2 输出
│   └── all_chunks.jsonl       ← 772 个 chunks，每行一个 JSON
│
└── faiss_index/               ← Step 3 输出
    ├── index.faiss
    └── metadata.jsonl
```

---

## 八、技术选型总结

| 组件 | 选型 | 理由 |
|------|------|------|
| 数据源 | DailyMed API v2 | 官方 FDA 说明书，免费、无需 API key |
| XML 解析 | Python `xml.etree.ElementTree` | 标准库，无额外依赖 |
| HTTP 请求 | `requests` + OpenSSL | macOS LibreSSL 有 TLS 相容性问题 |
| Embedding | OpenAI text-embedding-3-small | 效果好，费用极低（本次约 $0.004） |
| 向量库 | FAISS（本地） | 轻量、无服务器依赖，适合原型开发 |
| LLM | Anthropic Claude API | 已有接入，强推理能力 |
