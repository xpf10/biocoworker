# BioCoworker - 生物信息学 RNA-Seq 下游分析 AI 协同助手

BioCoworker 是一个类似于 Qoder/QoderWork 的智能生物信息学桌面助手，专为 RNA-Seq 下游数据分析设计。它结合了 React + Electron 桌面端展示，Python 快速数据统计分析引擎，以及基于 LangGraph 的 `deepagents` 框架，能够使用国内的大语言模型进行智能交互分析。

---

## 🌟 核心特性

1. **交互式 AI 聊天助手**:
   - 使用 `deepagents` 驱动，利用国内大模型（如通义千问 Qwen、DeepSeek 等）。
   - 具备 **数据汇总感知**、**一键差分表达分析**、**通路富集计算** 和 **基因详情检索** 等内置智能工具。
   
2. **差分表达分析 (DE Analysis)**:
   - 支持在后台运行 T-Test 与 Benjamini-Hochberg FDR 多重假设检验矫正。
   - 包含高互动性的 **Volcano 散点图**：支持 Log2FC 和 P-Adj 阈值滑块过滤，支持基因搜索和悬停悬浮框显示数据。
   - 包含 Z-Score 标准化的 **表达热图 (Expression Heatmap)**，动态展示差异最显著的 Top 25 基因在样本中的表达模式。

3. **PCA 样本聚类**:
   - 自动在后台进行 CPM (Counts Per Million) 变换与 Log2 转换，投影计算样本的主成分 PC1 和 PC2 坐标。
   - 互动展示样本的可视化聚类，方便科研人员评估生物学重复的紧密性。

4. **通路富集分析 (Pathway Enrichment)**:
   - 针对显著性差异表达基因，基于超几何分布算法 (Hypergeometric Test) 匹配背景生物通路数据库。
   - 直观展示通路富集的统计显著度折线/条形图，以及通路内重叠基因的列表详情。

5. **本地数据加载与测试**:
   - 支持一键生成包含负二项分布噪音的 1500 个基因、6 个样本（3 组对照，3 组处理）的模拟 RNA-Seq 数据。
   - 支持上传自定义的基因表达 Counts 矩阵文件与 Sample Design 表格。

---

## 📂 项目结构

```text
biocoworker/
├── backend/                  # Python 后端 API 与 Agent 模块
│   ├── main.py               # FastAPI 路由、文件上传与跨域配置
│   ├── agent.py              # 整合 deepagents + ChatOpenAI 及 Agent 工具
│   └── analysis.py           # 差异表达统计、PCA 变换、通路富集、Z-Score 计算核心算法
├── frontend/                 # React + Electron 桌面端
│   ├── src/
│   │   ├── App.tsx           # 包含 Volcano/PCA/Heatmap 的 SVG 绘制与聊天 UI 逻辑
│   │   ├── index.css         # Premium 轻量与深色自适应视觉规范
│   │   └── App.css           # 栅格及局部卡片响应式排版
│   ├── main.cjs              # Electron 主进程入口 (CJS)
│   ├── package.json          # 依赖配置与并发启动脚本
│   └── vite.config.ts        # Vite 打包配置 (包含相对基准路径支持)
├── pyproject.toml            # uv 项目依赖描述
├── run_backend.py            # 后端 FastAPI 启动入口
└── run_dev.ps1               # Windows PowerShell 一键开发启动脚本
```

---

## ⚙️ 环境依赖与准备

您需要安装以下环境：
- **Node.js** (v18+)
- **Python** (3.10+) 
- **uv** (Python 快速包管理器)

---

## 🚀 启动与开发

### 1. 自动启动（Windows 一键启动）
在项目根目录下打开 PowerShell 并运行以下脚本，即可并发启动后端 FastAPI 与前端 React + Electron：
```powershell
./run_dev.ps1
```

### 2. 手动分布启动

**第一步：启动 Python 后端**
```bash
# 激活 uv 环境并运行后端
uv run run_backend.py
```
*后端将在 `http://127.0.0.1:8989` 运行。API 交互文档可见 `http://127.0.0.1:8989/docs`。*

**第二步：运行前端桌面客户端**
```bash
cd frontend
npm run electron:dev
```
*这将启动 Vite 服务并由 Electron 加载，并在 Electron 内打开开发者工具 (DevTools) 方便调试。*

---

## 🇨🇳 国内大模型设置

在客户端的 **Settings** 面板中，您可以轻松配置国内的模型接口：
- **Model Name**: 例如 `qwen-plus`, `deepseek-chat`, `glm-4` 等。
- **API Base URL**: 模型厂商的 OpenAI 兼容端点，例如：
  - 阿里 DashScope: `https://dashscope.aliyuncs.com/compatible-mode/v1`
  - DeepSeek: `https://api.deepseek.com/v1`
  - 智谱 AI: `https://open.bigmodel.cn/api/paas/v4`
- **API Secret Key**: 您的模型提供商 API Key。

> [!NOTE]
> 如果您未配置任何 API 密钥，客户端仍可以完全离线运行 **所有的常规分析和可视化绘图功能**。在进行 AI Chat 时，Agent 会友好地给您提供本地调试状态下的提示与建议。
