# MedBuddy RAG 项目背景

## 项目目标
为 MedBuddy 慢病用药管理 app 构建 RAG 知识库，提升 AI 问诊对话的可信度。
目标用户是美国慢性病患者。

## 当前任务
1. 从 DailyMed 下载 15 种慢性病药品 FDA 说明书 XML
2. 解析 XML，按 section 切片（chunking）
3. Embedding + 存入向量数据库（FAISS 或 ChromaDB）
4. 接入现有 MedBuddy 前端问诊对话

## 药品列表
- metformin（tablet extended release）
- glipizide（tablet）
- dapagliflozin（tablet）
- sitagliptin（tablet）
- amlodipine（tablet）
- lisinopril（tablet）
- losartan（tablet）
- hydrochlorothiazide（tablet）
- atorvastatin（tablet）
- rosuvastatin（tablet）
- metoprolol（tablet）
- aspirin（tablet）
- levothyroxine（tablet）
- albuterol（aerosol）
- alendronate（tablet）

所有 XML 文件保存到 ./drug_labels/ 文件夹下。

## 技术选型
- Embedding: OpenAI text-embedding-3-small 或 BGE
- 向量库: FAISS（本地）
- LLM: Anthropic Claude API（已有）
