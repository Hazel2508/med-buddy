# 慢病用药小管家

帮助慢性病患者建立个人用药计划与复诊提醒的网页应用，内置基于 FDA 官方说明书的 RAG AI 问诊。

**在线访问：** `https://hazel2508.github.io/med-buddy/`

---

## 功能

- **今日用药** — 按早/中/晚/睡前展示用药清单，一键打卡
- **我的药箱** — 管理药品信息、追踪库存余量、低库存预警
- **复诊计划** — 记录就诊信息（医院、医嘱、用药调整），管理下次复诊日期
- **用药记录** — 月历打卡视图、7天/30天依从率统计
- **AI 问诊** — 基于 FDA 官方药品说明书的 RAG 问答，回答有据可查

数据存储在浏览器本地（localStorage），无需注册账号。

---

## 架构

```
前端 (GitHub Pages，纯静态)
      ↓ POST /api/chat
RAG 后端 (本地 / Render / Railway)
      ↓ BGE 向量检索 → FAISS → Claude API
      ↓ 返回中文回答 + 来源章节
前端渲染
```

AI 回答基于 15 种常见慢病药品（metformin、atorvastatin、amlodipine 等）的 FDA 官方说明书，通过 RAG 检索后交由 Claude 用中文作答。

---

## 本地运行（前端）

直接双击 `index.html` 用浏览器打开。  
AI 问诊功能需要同时启动后端服务（见下方）。

---

## 启动 RAG 后端

后端代码在 `~/Desktop/medbuddy/` 目录（与前端分开存放）。

### 环境要求

```bash
cd ~/Desktop/medbuddy
pip install fastapi uvicorn[standard] anthropic sentence-transformers faiss-cpu numpy
# 或直接：
pip install -r requirements_api.txt
```

### 运行

```bash
cd ~/Desktop/medbuddy
export ANTHROPIC_API_KEY="sk-ant-..."
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问 `http://localhost:8000/health` 验证服务正常。

后端启动后，打开 `index.html`，AI 问诊即可正常使用。

---

## 部署到 Render（免费）

将 AI 后端部署到云端，前端才能在任何设备上使用 AI 功能。

### 步骤

1. **准备后端仓库**
   ```bash
   # 把 medbuddy/ 目录推到一个新的 GitHub 仓库
   cd ~/Desktop/medbuddy
   git init && git add api.py query.py requirements_api.txt faiss_index/
   git commit -m "Add RAG API server"
   git remote add origin https://github.com/YOUR_USERNAME/medbuddy-api.git
   git push -u origin main
   ```
   > `bge_model/`（~127MB）不要推到 GitHub（超过限制）。  
   > Render 启动时会自动运行 `python download_model.py` 下载模型。

2. **在 Render 创建 Web Service**
   - 访问 [render.com](https://render.com) → New → Web Service
   - 连接上一步的仓库
   - Build Command: `pip install -r requirements_api.txt`
   - Start Command: `python download_model.py && uvicorn api:app --host 0.0.0.0 --port $PORT`
   - 添加环境变量：`ANTHROPIC_API_KEY = sk-ant-...`

3. **更新前端 API 地址**

   在 `index.html` 顶部找到：
   ```javascript
   const CHAT_API = 'http://localhost:8000/api/chat';
   ```
   改为 Render 给你的 URL：
   ```javascript
   const CHAT_API = 'https://your-medbuddy-api.onrender.com/api/chat';
   ```
   推送到 `main` 分支，GitHub Actions 会自动重新部署前端。

---

## 发布前端到 GitHub Pages

1. 将本仓库 push 到 GitHub
2. 进入仓库 **Settings → Pages**，Source 选择 `gh-pages` 分支，点击 Save
3. 等待约 1 分钟，访问 `https://hazel2508.github.io/med-buddy/`

每次向 `main` 分支推送代码，GitHub Actions 自动重新部署。

---

## 支持的药品（AI 问诊）

| 类别 | 药品 |
|------|------|
| 2型糖尿病 | metformin, glipizide, dapagliflozin, sitagliptin |
| 高血压 | amlodipine, lisinopril, losartan, hydrochlorothiazide, metoprolol |
| 高脂血症 | atorvastatin, rosuvastatin |
| 心血管 | aspirin |
| 甲状腺 | levothyroxine |
| 哮喘 | albuterol |
| 骨质疏松 | alendronate |
