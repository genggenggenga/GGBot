# GGBot Diagrams

架构图按版本分层，文件名在各版本间保持一致。

## 目录

| 目录 | 日期 | 用途 |
|---|---|---|
| `v0/` | 2026-08-05 | 初版设计及生成源文件 |
| `v1/` | 2026-08-07 | 第一版技术实现文档配图 |
| `v2/` | 2026-08-09 | 当前 v2 技术实现文档配图 |

`v0/source/` 保存初版绘图 JSON 和预览素材，不作为文档引用资源。

## 文件

| 编号 | 文件名 | 内容 |
|---|---|---|
| `01` | `system-architecture` | 系统整体架构 |
| `02` | `nlu-dialogue-state` | NLU 与 Dialogue State Tracking |
| `03` | `multi-agent-runtime` | Multi-Agent 路由与有界执行 |
| `04` | `tool-security` | 工具授权、确认与副作用安全 |
| `05` | `hybrid-rag` | Hybrid RAG 检索与回答链路 |
| `06` | `memory-persistence` | 记忆分层与持久化 |

每张正式图同时保存：

- `.svg`：可编辑源文件；
- `.png`：Markdown 文档引用的渲染版本。

新增版本时创建新的 `vN/` 目录并沿用相同文件名，不覆盖历史版本。
