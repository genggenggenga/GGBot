# GGBot Skills 文档

GGBot 启动时会从 `GGBOT_SKILLS_DIR` 读取 Skills，并在售后请求匹配时注入 `AfterSalesAgent` 的 ReAct prompt。Skills 适合维护业务处理规范、售后话术、审核边界和升级规则。

当前内置售后 Skill：

```text
skills/after_sales/common/SKILL.md  # 通用安全基线
skills/after_sales/refund/SKILL.md  # 退款申请
skills/after_sales/return/SKILL.md  # 退货申请
skills/after_sales/cancel/SKILL.md  # 取消订单
skills/after_sales/handoff/SKILL.md # 投诉与转人工
skills/after_sales/request/SKILL.md # 通用售后请求
```

## Skill 文件格式

推荐每个 Skill 使用独立目录，并将主文件命名为 `SKILL.md`：

```text
skills/<skill_name>/SKILL.md
```

文件顶部使用简单 front matter：

```markdown
---
name: 退款申请 SOP
description: 适用于 AfterSalesAgent 的退款申请处理规范
keywords:
agents: after_sales
intents: refund_request
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称，会出现在注入给模型的 prompt 中。
- `description`：简短说明，方便 `/skills` 接口排查。
- `keywords`：触发关键词，用户消息命中后才注入；多个关键词用英文逗号或中文逗号分隔均可。
- `agents`：适用 Agent，当前主运行时仅支持 `after_sales`。
- `intents`：适用意图，用于精确选择业务 SOP；留空表示对该 Agent 的所有意图生效。
- `enabled`：是否启用，支持 `true/false`。

## 编写要求

- 重要规则放在文档前半部分，因为过长内容会按 prompt 预算截断。
- 一类 Skill 只描述一类职责，不要把技术、账单、通用客服规则混在一个文件里。
- 必须包含“角色定位”“处理流程”“升级条件”“禁止事项”等稳定章节。
- 对用户隐私、支付、密码、验证码、API Key、Token 等敏感信息必须写明禁止收集或禁止公开。
- 对无法保证的事项使用保守措辞，例如“通常”“预计”“需要核验后确认”。
- 对需要人工、财务、二线技术处理的场景要明确写出升级条件。

## 热加载

修改 Skill 文件后，不需要重启服务，调用：

```bash
curl -X POST http://localhost:8000/skills/reload
```

查看加载结果和解析错误：

```bash
curl http://localhost:8000/skills
```
