---
knowledge_id: enterprise-customer-service
version: v1
version_seq: 1
status: published
business_domain: customer_service
product_line: ecommerce
effective_at: 2026-01-01T00:00:00Z
---

# 企业客服知识库 v1

<!-- GGKB:BEGIN type=faq id=refund.apply intent=refund_request business_domain=after_sales risk=normal -->
## 退款申请 FAQ

用户可以在订单详情页点击“申请售后”，选择退款原因并提交申请。
未发货订单可以直接申请仅退款；已发货订单需要先完成退货流程，仓库签收并验收通过后再退款。
退款审核通常在 1-3 个工作日内完成，审核通过后款项会在 5-7 个工作日内退回原支付账户。

同义问法：
- 怎么申请退款？
- 我要退款怎么办？
- 订单 ORD-1001 能不能退款？
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=refund.no_reason_7d intent=refund_request business_domain=after_sales policy_type=refund risk=normal -->
## 七天无理由退款规则

用户签收商品后 7 天内，商品未使用、未洗涤、吊牌和包装完整，可以申请七天无理由退货退款。
生鲜、定制商品、虚拟商品、拆封后影响二次销售的贴身用品不适用七天无理由。
如果商品存在质量问题，不受七天无理由限制，用户需要提供照片、视频或检测凭证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=refund.rejected_cases intent=refund_request business_domain=after_sales policy_type=refund risk=normal -->
## 退款拒绝场景

以下情况不得承诺直接退款：订单不存在、订单不属于当前用户、商品已超过售后期限且无质量凭证、用户要求退回非原支付账户、用户拒绝退货但要求已发货商品全额退款。
遇到上述情况时，客服应说明原因，并引导用户补充订单号、身份验证或质量凭证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=refund.sop.verify parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 退款处理流程 - 核验信息

第一步：客服需要核验订单号、用户身份、订单状态、支付状态和售后历史。
如果用户无法提供订单号，应先引导用户在订单列表查询，不得凭手机号直接透露订单信息。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=refund.sop.confirm parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 退款处理流程 - 确认动作

第二步：退款、取消订单、改地址、补发和赔付都属于写操作。
客服在执行写操作前必须向用户复述将要执行的动作、订单号、金额或地址，并获得明确确认。
用户只表达“帮我看看”“可以吗”不等于确认执行。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=section_parent id=refund.sop.parent parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 完整退款处理流程

客服处理退款时，先核验订单号、用户身份、订单状态、支付状态和售后历史。
未发货订单可申请仅退款；已发货订单需判断是否签收、是否需要退货、是否存在质量问题。
涉及退款执行时，客服必须先向用户复述动作和关键信息，并获得明确确认。
如果订单不属于当前用户、无法验证身份、或用户要求退回非原支付账户，应拒绝执行并转人工复核。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=table id=logistics.delay_compensation intent=logistics_delay business_domain=logistics policy_type=compensation risk=normal -->
## 物流延迟补偿标准

| 场景 | 判定条件 | 补偿标准 | 限制 |
|---|---|---|---|
| 标准配送延迟 | 超过承诺送达时间 3-5 天 | 10 元无门槛优惠券 | 每个订单仅补偿一次 |
| 严重配送延迟 | 超过承诺送达时间 6 天及以上 | 20 元无门槛优惠券 | 需人工审核 |
| 偏远地区延迟 | 地址属于偏远地区且延迟不超过 3 天 | 不补偿 | 需解释偏远地区时效 |
| 不可抗力延迟 | 台风、洪水、交通管制等官方公告原因 | 不自动补偿 | 可转人工安抚 |
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=faq id=coupon.expired intent=coupon_issue business_domain=membership risk=normal -->
## 优惠券过期 FAQ

优惠券过期后默认不能补发，也不能延长有效期。
如果优惠券因系统故障、物流严重延迟或客服错误承诺导致无法使用，可以提交人工复核。
复核通过后，只能补发同等面额或等价权益，不得承诺现金补偿。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=invoice.issue intent=invoice_request business_domain=order_payment policy_type=invoice risk=normal -->
## 发票开具规则

用户可以在订单完成后 30 天内申请电子发票。
发票抬头、税号和邮箱需要由用户本人提供；客服不得替用户猜测或编造发票信息。
已开具发票需要修改时，必须先作废原发票，再重新提交开票申请。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=privacy.protect_order intent=privacy_protection business_domain=safety risk=high -->
## 隐私与越权查询规则

客服不得向用户透露他人的手机号、身份证号、收货地址、支付账号、订单明细或物流轨迹。
如果用户请求查询他人订单，必须拒绝，并引导订单本人完成身份验证。
如果用户只提供手机号但未完成验证，不得返回该手机号关联的订单列表。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=write.confirmation_required intent=write_action_confirmation business_domain=safety risk=high -->
## 写操作确认规则

退款、取消订单、修改地址、补发商品、发放补偿、关闭售后单都属于写操作。
执行写操作前必须获得用户明确确认，确认内容需要覆盖动作、订单号和关键金额或地址。
缺少确认时，只能说明将要执行的操作并请求确认，不得直接调用写工具。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=human.handoff intent=human_handoff business_domain=customer_service policy_type=sla risk=normal -->
## 人工升级规则

以下情况应升级人工：用户连续两次否认机器人理解、涉及投诉升级、订单金额超过 2000 元、物流丢件、质量纠纷证据不一致、系统查询失败但用户要求立即处理。
升级人工时，客服应总结已核验的信息、用户诉求、订单号、当前阻塞原因，避免用户重复描述。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=faq id=order.cancel_unpaid intent=order_cancel business_domain=order_payment risk=normal -->
## 未支付订单取消 FAQ

未支付订单可以由用户在订单详情页自行取消，取消后库存和优惠券会按系统规则释放。
如果订单已超过支付时限，系统会自动关闭，客服无需手动处理。
如果用户询问“为什么不能继续支付”，客服应解释订单已关闭，需要重新下单。

同义问法：
- 未付款订单怎么取消？
- 订单超时了还能支付吗？
- 订单没付款会自动关吗？
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=order.address_change intent=address_change business_domain=order_payment policy_type=order risk=normal -->
## 修改收货地址规则

订单未发货前，用户可以申请修改收货地址；已发货订单原则上不支持修改地址。
如果物流公司支持中途改派，客服只能协助提交改派申请，不得承诺一定成功。
修改地址属于写操作，客服必须在提交前复述新地址、收件人和手机号，并获得用户明确确认。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=table id=return.shipping_fee intent=return_shipping_fee business_domain=after_sales policy_type=refund risk=normal -->
## 退货运费承担标准

| 场景 | 运费承担方 | 需要凭证 | 说明 |
|---|---|---|---|
| 七天无理由退货 | 用户承担 | 否 | 商品需保持完好且不影响二次销售 |
| 商品质量问题 | 平台或商家承担 | 是 | 需提供照片、视频或检测凭证 |
| 发错货或漏发 | 平台或商家承担 | 是 | 需提供实物照片和包裹面单 |
| 用户填错地址导致退回 | 用户承担 | 否 | 重新发货可能需要补运费 |
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=quality_issue.sop.collect_evidence parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 质量问题处理流程 - 收集凭证

第一步：客服需要引导用户提供问题商品照片、外包装照片、订单号和问题描述。
如果是电子产品、家电或高价值商品，还需要用户提供开箱视频或检测报告。
客服不得在未核验证据前承诺全额退款、补发或赔付。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=quality_issue.sop.judge_solution parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 质量问题处理流程 - 判断方案

第二步：客服根据凭证判断处理方案。
轻微瑕疵可优先协商部分补偿；影响使用的问题应进入退换货流程；存在安全风险的问题必须升级人工。
如果用户证据不足，客服应说明需要补充哪些材料，而不是直接拒绝。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=section_parent id=quality_issue.sop.parent parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 完整质量问题处理流程

质量问题处理需要先收集订单号、商品照片、外包装照片、问题描述和必要的视频或检测报告。
客服应根据问题严重程度选择部分补偿、退货退款、换货、补发或人工升级。
涉及退款、补发、赔付等动作时，必须遵守写操作确认规则。
证据不足时，应明确告知需要补充的材料，不得直接编造判断结论。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=account.security_verification intent=account_security business_domain=safety risk=high -->
## 账号安全验证规则

客服不得索要用户登录密码、短信验证码、支付密码或银行卡完整卡号。
如果用户反馈账号被盗、异常登录或订单被他人操作，应立即建议用户修改密码、退出其他设备，并转人工安全队列。
客服只能引导用户通过官方验证流程找回账号，不得绕过身份验证直接修改账号资料。
<!-- GGKB:END -->
