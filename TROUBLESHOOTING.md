# MedBuddy 开发问题与解决记录

> 此文档记录项目全过程遇到的问题和解决方法，持续更新。
> 格式：问题描述 → 根因 → 解决方案 → 教训

---

## Step 1 — DailyMed XML 下载

### 问题 1：v2 API zip 下载端点失效
**现象：** `GET /services/v2/spls/{setid}.zip` 返回 302，重定向到 DailyMed 首页 HTML，ZipFile 解析报 `File is not a zip file`。

**根因：** DailyMed 已废弃 v2 REST API 的 zip 下载路径，该 endpoint 静默 302 到首页，不返回 4xx 错误，难以察觉。

**解决方案：** 改用非 API 的 web 下载端点：
```
https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm?setId={setid}
```
该端点返回 `Content-Type: application/zip`，response header `X-DAILYMED-LABEL-LAST-UPDATED` 即说明书最后更新日期。

**教训：** API 文档说的端点不一定还在运行。遇到下载返回 HTML 时，先检查 status code 和 final URL（follow redirect 后的地址）。

---

### 问题 2：macOS LibreSSL 与 DailyMed TLS 不相容
**现象：** 用 `subprocess` 调 `curl` 下载，约 50% 请求报 `LibreSSL SSL_connect: SSL_ERROR_SYSCALL` (curl exit 35) 或 `Connection timed out` (exit 28)，交替出现，整批下载几乎全部失败。

**根因：** macOS 系统 curl 使用 LibreSSL（Apple 维护），与 DailyMed 服务器的 TLS 握手有间歇性相容性问题；conda 环境的 Python `requests` 库使用 OpenSSL，同样网络环境下完全稳定。

**解决方案：** 弃用 `subprocess + curl`，改用 Python `requests` 库（conda 自带 OpenSSL）：
```python
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_retry = Retry(total=4, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_retry))
```

**教训：** macOS 脚本若需调用 HTTPS API，优先用 Python `requests` 而非系统 `curl`；系统 LibreSSL 版本落后，容易有 TLS 握手问题。

---

### 问题 3：搜索结果混入复方药说明书
**现象：** 搜索 `drug_name=metformin` + 筛选 TABLET，返回 setid 对应说明书标题为 `GLYBURIDE AND METFORMIN HYDROCHLORIDE TABLET, FILM COATED`（格列本脲+二甲双胍复方），而非纯二甲双胍说明书。

**根因：** DailyMed 搜索以 `drug_name` 为关键字匹配，复方药包含此关键字也会返回；而复方药近期更新较多，按 `published_date` 排序后排在最前。

**解决方案：** 在筛选 TABLET/AEROSOL 之后，额外要求说明书 title 以目标药名开头（大小写不敏感），排除复方：
```python
single = [m for m in matches if m["title"].upper().startswith(drug_name.upper())]
candidates = single if single else matches  # 没有单成分时回退到全部
```

**教训：** DailyMed 搜索不区分单成分与复方；为知识库下载说明书时必须主动过滤，否则 RAG 回答时药物信息会混淆。

---

### 问题 4：API 字段名与文档描述不符
**现象：** 按 DailyMed 文档描述预期有 `updated_date` 字段；实际 v2 API 搜索响应只有 `published_date`，无 `updated_date`。

**根因：** DailyMed API 文档描述与实际响应字段有出入；`published_date` = SPL 最后发布日期，即实际意义上的"更新日期"。真正的更新时间戳在 `downloadzipfile.cfm` 响应 header 的 `X-DAILYMED-LABEL-LAST-UPDATED` 中。

**解决方案：** 使用 `published_date` 作为排序依据（parse 格式：`"Jun 05, 2026"` → `datetime.strptime(s, "%b %d, %Y")`）；若需精确更新时间，从下载 response header 取 `X-DAILYMED-LABEL-LAST-UPDATED`。

**教训：** 对 API 字段名做假设之前先打印实际 JSON 响应确认。

---

### 问题 5：macOS 扩展属性 com.apple.macl 阻塞文件写入
**现象：** 代码编辑工具报 `EPERM: operation not permitted` 尝试修改 `download_labels.py`，但 `ls -la` 显示文件属主和权限正常（`-rw-r--r-- ever staff`）。

**根因：** 文件有 macOS 强制访问控制扩展属性 `com.apple.macl`（`ls` 输出中 `@` 符号标识），限制只有创建该文件的进程/app 才能修改。

**解决方案：** 用 `xattr` 移除该属性即可恢复正常写入：
```bash
xattr -d com.apple.macl <filepath>
```

**教训：** macOS 上 EPERM 不一定是 UNIX 权限问题，要同时检查 `xattr -l` 看扩展属性；`@` 符号在 `ls -la` 输出中是提示有扩展属性的信号。

---

## Step 2 — XML 解析与 Chunking

### 问题 6：解析返回 0 个 chunk
**现象：** `parse_chunks.py` 对 15 个 XML 文件全部输出 0 chunks，没有报错。

**根因：** SPL XML 的 `<section>` 不是 `<document>` 的直接子节点，而是深嵌在 `component → structuredBody → component → section` 包装层级中。脚本只遍历直接子节点找 `<section>` 标签，自然找不到。

**解决方案：** 修改递归逻辑：遇到非 `<section>` 标签时，继续向下递归寻找 section，而不是 skip：
```python
for child in node:
    if child.tag != f"{{{NS}}}section":
        yield from iter_chunks(child, ancestor_titles)  # 继续下探
        continue
    # ... 处理 section ...
```

**教训：** 解析有命名空间的 HL7 XML 前，先用 `root.tag` 确认命名空间，再用 `root.iter()` 或打印前几层层级确认实际 XML 结构，不要假设元素在哪个深度。

---

### 问题 7：部分 section 显示为 "(untitled)"
**现象：** 510 个 chunk 中，多个 section 路径显示为 `(untitled)`，包括 albuterol 的 15,134 chars 大块，metoprolol 的多个主要章节。

**根因：** SPL 规范允许 `<section>` 不含 `<title>` 子元素，section 身份仅由 `<code code="LOINC_CODE">` 标识。例如 Patient Information (LOINC 42230-3)、Medication Guide (42231-1) 通常无 `<title>`。

**解决方案：** 建立 LOINC code → 标准英文名称的映射表作兜底，当 `<title>` 元素不存在时使用映射名称：
```python
LOINC_NAMES = {
    "42230-3": "Patient Information",
    "42231-1": "Medication Guide",
    "34090-1": "Clinical Pharmacology",
    # ... 共 25+ 个常用 LOINC 代码
}
title = LOINC_NAMES.get(code, "") if title_el is None else clean_text(title_el)
```
修复后 0 个 `(untitled)` chunk。

**教训：** DailyMed SPL 的 section 标题有时只在 LOINC code 中隐含，不能仅靠 `<title>` 标签获取章节语义。常见章节 LOINC 代码应作为常量维护在代码中。

---

### 问题 8：原始 section 文本过大，不适合 embedding
**现象：** 510 个 chunk 中有 44 个超过 4000 chars，最大 28,726 chars（dapagliflozin 临床研究章节）。OpenAI text-embedding-3-small token 上限 8191（≈30k chars），极端情况下可能截断。

**根因：** SPL 以章节为最小单位，临床研究（Section 14）、药代动力学（Section 12.3）等章节本身文字量极大，未做进一步细分。

**解决方案：** 对超过 2000 chars 的 chunk 按句子边界（`(?<=[.!?])\s+`）拆分，相邻 sub-chunk 保留 200 chars 重叠窗口保证上下文连贯性：
```python
def split_large_chunk(text, max_chars=2000, overlap=200):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 按 max_chars 分组，组间保留 overlap 字符
    ...
```
拆分后共 772 chunks，仅 5 个略超 2000 chars（最大 4516），均为超长医学术语单句，在 embedding 模型 token 限制内。

**教训：** RAG chunking 策略：句子边界 > 段落边界 > 固定字符截断。句子边界拆分对医学文本效果好；纯固定长度截断会破坏语义完整性。重叠窗口（overlap）对跨 chunk 的问答查询很重要。

---

## Step 3 — Embedding + 向量数据库

### 问题 9：sentence-transformers 用 httpx 下载模型，代理返回 503
**现象：** `SentenceTransformer("BAAI/bge-small-en-v1.5")` 抛 `httpcore.ProxyError: 503 Service Unavailable`，模型无法下载。

**根因：** `sentence-transformers` 内部调用 `huggingface_hub`，后者使用 `httpx` 下载文件。`httpx` 自动读取 macOS 系统代理（`127.0.0.1:1082`），但该代理对 huggingface.co 返回 503（代理客户端未启动或路由问题）。`requests` 虽然也读系统代理，但通过代理能访问部分 CDN；`httpx` 对某些路由失败更严格。

**解决方案：** 单独写 `download_model.py`，用 `requests` + `trust_env=False`（绕过系统代理）直连 `hf-mirror.com`（HuggingFace 中国镜像）下载所有模型文件到本地目录 `./bge_model/`，再通过 `SentenceTransformer('./bge_model')` 从本地加载：
```python
session = requests.Session()
session.trust_env = False  # 绕过系统代理，直连 hf-mirror.com
```

**教训：** 在中国网络环境下使用需要下载模型的库，必须提前处理代理问题；`trust_env=False` 可以让 `requests` 绕开系统代理直连；HuggingFace 中国镜像 `hf-mirror.com` 可替代官方源。

---

### 问题 10：HuggingFace XET 下载协议卡住
**现象：** 设置 `HF_ENDPOINT=https://hf-mirror.com` 后，模型下载卡在 100MB/133MB，进程空跑不动，XET 日志显示反复 "connection struggling"，最终报 `TimedOut`。

**根因：** 新版 `huggingface_hub` 默认使用 XET 协议（hf_xet）从 AWS CDN（`us.aws.cdn.hf.co`）并发下载，XET CDN 在当前网络环境不稳定。虽然设置了 `HF_HUB_DISABLE_XET=1` 禁用 XET，但退回到的标准 HTTPS 下载流通过代理也不稳定。

**解决方案：** 同问题 9——完全绕开 `huggingface_hub` 的下载机制，用自定义 `requests` 脚本下载。

**教训：** `HF_HUB_DISABLE_XET=1` 可以禁用 XET，但如果系统代理也有问题，仍然无济于事；根本解是直连可用的镜像源。

---

### 问题 11：macOS com.apple.macl 阻塞模块导入（再次）
**现象：** 运行新创建的 Python 脚本时报 `PermissionError: [Errno 1] Operation not permitted`，位置在 `importlib._bootstrap._path_importer_cache`，即 Python 尝试 import 任何模块时就失败。

**根因：** 由 Claude Code 的 Write 工具创建的文件自动获得 `com.apple.macl` 扩展属性（macOS 强制访问控制），限制只有创建该文件的进程可以读写。Python 解释器在加载模块时需要扫描目录路径，如果脚本本身有 macl 属性，macOS sandbox 会阻止相关路径操作。

**解决方案：** 每次新建脚本后执行：
```bash
xattr -d com.apple.macl <filename>.py
```
或批量清除（需要对每个文件单独操作，目录级别的 `xattr -rc` 会因 EPERM 失败）。

**教训：** 凡是由 Write/Edit 工具新建的文件，运行前都要先清 `com.apple.macl`；这个问题会在项目全程反复出现（每次新建文件都要处理）。考虑在 `.zshrc` 加 alias 或用 `umask` 解决。

---

## Step 4 — 接入前端对话

### 问题 12：BGE query embedding 需要特定前缀
**现象：** 用 BGE 直接编码查询文本，检索结果相关性较低，不如预期。

**根因：** BAAI/bge-small-en-v1.5 在检索任务中，query 侧需加前缀 `"Represent this sentence: "` 以触发检索模式（passage 侧无需前缀）。不加前缀时，query 向量的分布与 document 向量不在同一语义空间，相似度得分偏低。

**解决方案：** `embed_query()` 中对问题加前缀：
```python
prefixed = f"Represent this sentence: {question}"
vec = model.encode([prefixed], normalize_embeddings=True)
```

**教训：** 使用 BGE 系列模型做检索时，必须区分 query 侧和 passage 侧的编码方式，参考模型文档中 "query instruction"。

---

### 问题 13：drug_filter 命中率低（过滤后无结果）
**现象：** 传入 `--drug metformin` 时，FAISS 搜索 top-5 结果全是其他药品，过滤后返回空列表。

**根因：** FAISS 只按全局相似度排序，某个问题的 top-5 可能恰好全属于相似度更高的其他药品 chunk。

**解决方案：** 开启 drug_filter 时，先搜索更大范围（`top_k * 6`），再按 drug 名过滤取前 top_k：
```python
search_k = top_k * 6 if drug_filter else top_k
scores, indices = index.search(q_vec, search_k)
```

**教训：** 向量搜索 + 元数据过滤组合时，总是先放大搜索范围再过滤，否则过滤后结果数不足。
