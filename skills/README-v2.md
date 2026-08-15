# GGBot Skills 文档 v2

GGBot 从 `GGBOT_SKILLS_DIR` 加载 Skill。当前主运行时只把匹配的 Skill 注入 AfterSalesAgent 的 ReAct Planner；Skill 不再全局拼入 MemoryContext 或 KnowledgeAgent 的检索问题。

## 当前文件

```text
skills/after_sales/common/SKILL.md  # 所有售后意图共享的安全基线
skills/after_sales/refund/SKILL.md  # refund_request
skills/after_sales/return/SKILL.md  # return_request
skills/after_sales/cancel/SKILL.md  # cancel_order
skills/after_sales/handoff/SKILL.md # complaint, escalation
skills/after_sales/request/SKILL.md # request
```

旧 `billing_support` 已删除，由 `after_sales` 承接退款、退货、取消订单、投诉和转人工 SOP。

## 作用边界

售后 Skill 可以影响：

- 只读核验顺序；
- 信息不足时的澄清策略；
- 回复措辞和业务阶段表达；
- RPC 异常后的保守建议。

售后 Skill 不能影响：

- Agent 工具白名单；
- ToolSpec 的 READ/WRITE 分类；
- PendingAction 与用户确认门禁；
- RPC 返回的资格、金额和状态；
- action_id 幂等与参数指纹；
- 身份认证和资源授权。

发生冲突时，以系统 Prompt、代码安全规则和 RPC 事实为准。

## Skill 文件格式

推荐每个 Skill 使用独立目录：

```text
skills/<skill_name>/SKILL.md
```

当前 front matter 支持：

```markdown
---
name: 退款申请 SOP
description: 退款申请的资格核验、金额说明、确认与提交规范
keywords:
agents: after_sales
intents: refund_request
version: 2
eval_cases: refund_request
enabled: true
---
```

字段说明：

- `name`：Skill 展示名称。
- `description`：`/skills` 中的简短说明。
- `keywords`：可选消息关键词；为空时由 Agent 和 Intent 决定匹配。
- `agents`：适用 Agent，主运行时使用 `after_sales`。
- `intents`：允许注入的业务意图。
- `version`：运营版本。
- `eval_cases`：建议关联的回归场景。
- `enabled`：是否启用。

## 匹配过程

```text
Router
  -> AfterSalesAgent.execute(goal)
  -> SkillManager.prompt_for(
       message,
       agent_type="after_sales",
       intent=goal
     )
  -> ReActPlanner
       [当前领域职责]
       [售后 Skill 软策略]
```

匹配规则会组合 `common` 与当前意图的专属 SOP。例如退款请求加载
`售后通用安全基线 + 退款申请 SOP`；投诉请求加载
`售后通用安全基线 + 投诉与转人工 SOP`。OrderAgent、LogisticsAgent 和
KnowledgeAgent 当前不消费 Skill。

## 编写要求

- 一份 Skill 只描述一类领域 SOP。
- 开头明确适用边界和不能覆盖的硬规则。
- 金额、状态、资格和编号必须沿用 RPC 结果。
- 写操作只允许建议待确认动作。
- 区分“申请已创建”“审核通过”“资金到账”。
- RPC 失败时不得补造事实或承诺结果。
- 不收集密码、验证码、支付密码、完整银行卡号、Token 或私钥。
- 修改后必须覆盖 Skill 匹配、Planner 选择和确认门禁测试。

## 热加载

```bash
curl -X POST http://localhost:8000/skills/reload
curl http://localhost:8000/skills
```

热加载只刷新 SkillManager 内容，不改变 ToolRegistry 或运行时权限。
