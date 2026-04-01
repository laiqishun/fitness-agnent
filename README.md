# Speediance AI 私教智能体

基于 LangGraph 的多意图 AI 健身私教，支持问答、课程推荐、提醒设置、饮食/运动记录和个性化运动计划生成。

## 架构概览

```
用户消息
  │
  ▼
query_rewriter   ── Qwen-Turbo 查询改写 + 扩展
  │
  ▼
intent_classifier ── QwenPlus 意图识别（5类意图）
  │
  ├─► qa_node              ── RAG + QwenPlus 问答
  ├─► course_recommendation ── 向量检索 + gte-rerank 推荐
  ├─► reminder_node        ── 时间解析 + App API 上报
  ├─► diet_exercise_recorder── 营养/消耗估算 + 数据库记录
  └─► workout_plan_generator── 构思→执行→反思 3轮循环
  │
  ▼
save_message（持久化）→ END
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 智能体框架 | LangGraph 0.2+ |
| 主力模型 | Qwen-Plus（通义千问，OpenAI 兼容接口）|
| 轻量模型 | Qwen-Turbo（查询改写） |
| 向量模型 | text-embedding-v3（1536维）|
| 重排序 | gte-rerank |
| 向量数据库 | PostgreSQL + pgvector |
| 会话记忆 | Redis（短期）+ PostgreSQL checkpointer（长期）|
| Web 框架 | FastAPI |
| ORM | SQLAlchemy 2.0 异步 |

## 快速开始

### 1. 环境准备

```bash
# 安装 uv（包管理工具）
pip install uv

# 克隆项目并安装依赖
cd fitness_agent
uv sync
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写必要配置：
# - DASHSCOPE_API_KEY
# - POSTGRES_PASSWORD
# - REDIS_PASSWORD（如有）
```

### 3. 初始化数据库

```bash
# 确保 PostgreSQL 已启动并安装 pgvector 扩展
psql -U postgres -c "CREATE DATABASE fitness_agent;"
psql -U postgres -d fitness_agent -f sql/schema.sql
```

### 4. 启动服务

```bash
# 开发模式（热重载）
uv run hatch run dev

# 或直接启动
uv run python -m fitness_agent.main
```

服务默认运行在 `http://localhost:8000`，Swagger 文档：`http://localhost:8000/docs`

## API 接口

### 发送消息
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "message": "帮我制定一个4周增肌计划"
  }'
```

### 创建会话
```bash
curl -X POST "http://localhost:8000/api/v1/sessions?user_id=user_001" \
  -H "Content-Type: application/json" \
  -d '{"title": "增肌计划对话"}'
```

### 获取历史
```bash
curl "http://localhost:8000/api/v1/sessions/{session_id}/history?user_id=user_001"
```

## 运行测试

```bash
uv run hatch run test
```

## 项目结构

```
fitness_agent/
├── pyproject.toml           # 项目配置 & 依赖
├── .env.example             # 环境变量示例
├── sql/
│   └── schema.sql           # PostgreSQL 完整 Schema
├── src/fitness_agent/
│   ├── config.py            # pydantic-settings 配置管理
│   ├── main.py              # FastAPI 应用入口
│   ├── models/
│   │   ├── database.py      # SQLAlchemy ORM 模型
│   │   └── schemas.py       # Pydantic 请求/响应 Schema
│   ├── graph/
│   │   ├── state.py         # AgentState 定义
│   │   ├── graph.py         # LangGraph 图组装
│   │   └── nodes/           # 各功能节点
│   ├── rag/
│   │   ├── embedder.py      # text-embedding-v3 向量化
│   │   ├── retriever.py     # 混合检索（向量+全文 RRF）
│   │   └── reranker.py      # gte-rerank 重排序
│   ├── memory/
│   │   ├── redis_short_term.py  # Redis 短期记忆
│   │   └── pg_checkpointer.py   # PG 持久化 checkpointer
│   ├── services/
│   │   ├── app_api.py       # App 后端接口调用
│   │   └── session_summarizer.py # 会话摘要（模糊记忆）
│   └── api/
│       └── routes.py        # FastAPI 路由
└── tests/
    └── test_graph.py        # 单元 + 集成测试
```

## 意图说明

| 意图 | 触发示例 | 处理节点 |
|------|---------|---------|
| `qa` | "俯卧撑怎么做" | qa_node（RAG）|
| `course_recommendation` | "推荐减脂课程" | course_recommendation |
| `set_reminder` | "明天8点提醒我健身" | reminder_node |
| `record_diet_exercise` | "刚跑步30分钟" | diet_exercise_recorder |
| `generate_workout_plan` | "制定增肌计划" | workout_plan_generator（Think→Execute→Reflect）|

## 记忆机制

- **短期记忆**：Redis 存储最近 20 条消息，TTL 1小时
- **长期记忆**：PostgreSQL checkpointer 持久化对话状态
- **模糊记忆**：每 20 条消息触发一次摘要，压缩细节保留关键事实
- **用户画像**：从摘要中自动提取并更新画像（体重、目标、伤病等）
