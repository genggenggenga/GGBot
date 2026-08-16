---
knowledge_id: enterprise-customer-service
version: v2
version_seq: 2
status: published
business_domain: customer_service
product_line: ecommerce
effective_at: 2026-03-01T00:00:00Z
---

# 企业客服知识库 v2

<!-- GGKB:BEGIN type=faq id=refund.apply intent=refund_request business_domain=after_sales risk=normal -->
## 退款申请 FAQ

用户可以在订单详情页点击“申请售后”，选择退款原因并提交申请。
未发货订单可以直接申请仅退款；已发货订单需要先完成退货流程，仓库签收并验收通过后再退款。
退款审核通常在 1 个工作日内完成，审核通过后款项会在 3-5 个工作日内退回原支付账户。
如果用户属于金卡或黑卡会员，客服可以提示其享受优先审核权益，但不得承诺即时到账。

同义问法：
- 怎么申请退款？
- 我要退款怎么办？
- 订单 ORD-1001 能不能退款？
- 退款多久能到账？
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=refund.no_reason_15d intent=refund_request business_domain=after_sales policy_type=refund risk=normal -->
## 十五天无理由退款规则

自 v2 起，普通商品支持签收后 15 天内无理由退货退款。
商品需未使用、未洗涤、吊牌和包装完整，不影响二次销售。
生鲜、定制商品、虚拟商品、拆封后影响二次销售的贴身用品仍不适用无理由退货。
如果商品存在质量问题，不受十五天无理由限制，用户需要提供照片、视频或检测凭证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=refund.rejected_cases intent=refund_request business_domain=after_sales policy_type=refund risk=normal -->
## 退款拒绝场景

以下情况不得承诺直接退款：订单不存在、订单不属于当前用户、商品已超过售后期限且无质量凭证、用户要求退回非原支付账户、用户拒绝退货但要求已发货商品全额退款。
如果用户存在异常售后频次、疑似恶意退货或批量薅补偿，客服不得自行拒绝，应升级人工风控复核。
遇到上述情况时，客服应说明原因，并引导用户补充订单号、身份验证或质量凭证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=refund.sop.verify parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 退款处理流程 - 核验信息

第一步：客服需要核验订单号、用户身份、订单状态、支付状态、售后历史和会员等级。
如果用户无法提供订单号，应先引导用户在订单列表查询，不得凭手机号直接透露订单信息。
如果系统查询失败，但用户能提供支付凭证，应升级人工而不是直接拒绝。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=refund.sop.confirm parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 退款处理流程 - 确认动作

第二步：退款、取消订单、改地址、补发、赔付和关闭售后单都属于写操作。
客服在执行写操作前必须向用户复述将要执行的动作、订单号、金额、地址或权益内容，并获得明确确认。
用户只表达“帮我看看”“可以吗”“你处理吧”不等于确认执行。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=section_parent id=refund.sop.parent parent_id=refund_sop intent=refund_request business_domain=after_sales risk=normal -->
## 完整退款处理流程

客服处理退款时，先核验订单号、用户身份、订单状态、支付状态、售后历史和会员等级。
未发货订单可申请仅退款；已发货订单需判断是否签收、是否需要退货、是否存在质量问题。
涉及退款执行时，客服必须先向用户复述动作和关键信息，并获得明确确认。
如果订单不属于当前用户、无法验证身份、用户要求退回非原支付账户、或命中异常售后频次，应拒绝直接执行并转人工复核。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=table id=logistics.delay_compensation intent=logistics_delay business_domain=logistics policy_type=compensation risk=normal -->
## 物流延迟补偿标准

| 场景 | 判定条件 | 补偿标准 | 限制 |
|---|---|---|---|
| 标准配送延迟 | 超过承诺送达时间 2-4 天 | 15 元无门槛优惠券 | 每个订单仅补偿一次 |
| 严重配送延迟 | 超过承诺送达时间 5 天及以上 | 30 元无门槛优惠券 | 需人工审核 |
| 生鲜订单延迟 | 生鲜商品超过承诺送达时间 1 天 | 订单实付金额 10% 优惠券 | 最高 30 元 |
| 偏远地区延迟 | 地址属于偏远地区且延迟不超过 3 天 | 不补偿 | 需解释偏远地区时效 |
| 不可抗力延迟 | 台风、洪水、交通管制等官方公告原因 | 不自动补偿 | 可转人工安抚 |
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=faq id=coupon.expired intent=coupon_issue business_domain=membership risk=normal -->
## 优惠券过期 FAQ

优惠券过期后默认不能补发，也不能延长有效期。
自 v2 起，只有系统故障、客服错误承诺或平台主动召回活动导致无法使用时，才可以提交人工复核。
物流延迟不再自动作为优惠券补发原因，但严重物流延迟可以按物流延迟补偿标准处理。
复核通过后，只能补发同等面额或等价权益，不得承诺现金补偿。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=invoice.issue intent=invoice_request business_domain=order_payment policy_type=invoice risk=normal -->
## 发票开具规则

用户可以在订单完成后 60 天内申请电子发票。
发票抬头、税号和邮箱需要由用户本人提供；客服不得替用户猜测或编造发票信息。
已开具发票需要修改时，必须先作废原发票，再重新提交开票申请。
企业发票金额必须与订单实付金额一致，优惠券、积分抵扣和运费减免不得重复开票。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=faq id=order.cancel_unpaid intent=order_cancel business_domain=order_payment risk=normal -->
## 未支付订单取消 FAQ

未支付订单可以由用户在订单详情页自行取消，取消后库存会立即释放。
如果订单已使用限量优惠券，优惠券释放可能存在 5 分钟延迟。
如果订单已超过支付时限，系统会自动关闭，客服无需手动处理。
如果用户询问“为什么不能继续支付”，客服应解释订单已关闭，需要重新下单。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=order.address_change intent=address_change business_domain=order_payment policy_type=order risk=normal -->
## 修改收货地址规则

订单未发货前，用户可以申请修改收货地址。
已发货订单仅在物流公司支持改派时可提交申请，客服不得承诺一定成功。
生鲜、同城即时配送、跨境订单发货后不支持修改地址，只能建议用户联系配送员或等待包裹退回。
修改地址属于写操作，客服必须在提交前复述新地址、收件人和手机号，并获得用户明确确认。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=table id=return.shipping_fee intent=return_shipping_fee business_domain=after_sales policy_type=refund risk=normal -->
## 退货运费承担标准

| 场景 | 运费承担方 | 需要凭证 | 说明 |
|---|---|---|---|
| 十五天无理由退货 | 用户承担 | 否 | 商品需保持完好且不影响二次销售 |
| 商品质量问题 | 平台或商家承担 | 是 | 需提供照片、视频或检测凭证 |
| 发错货或漏发 | 平台或商家承担 | 是 | 需提供实物照片和包裹面单 |
| 平台召回商品 | 平台承担 | 否 | 以召回公告为准 |
| 用户填错地址导致退回 | 用户承担 | 否 | 重新发货可能需要补运费 |
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=quality_issue.sop.collect_evidence parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 质量问题处理流程 - 收集凭证

第一步：客服需要引导用户提供问题商品照片、外包装照片、订单号、问题描述和收货时间。
如果是电子产品、家电、高价值商品或食品安全问题，还需要用户提供开箱视频、检测报告或批次信息。
客服不得在未核验证据前承诺全额退款、补发或赔付。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=sop id=quality_issue.sop.judge_solution parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 质量问题处理流程 - 判断方案

第二步：客服根据凭证判断处理方案。
轻微瑕疵可优先协商部分补偿；影响使用的问题应进入退换货流程；食品安全、用电安全、儿童用品安全问题必须升级人工。
如果用户证据不足，客服应说明需要补充哪些材料，而不是直接拒绝。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=section_parent id=quality_issue.sop.parent parent_id=quality_issue_sop intent=quality_issue business_domain=after_sales risk=normal -->
## 完整质量问题处理流程

质量问题处理需要先收集订单号、商品照片、外包装照片、问题描述、收货时间和必要的视频、检测报告或批次信息。
客服应根据问题严重程度选择部分补偿、退货退款、换货、补发、平台召回处理或人工升级。
涉及退款、补发、赔付等动作时，必须遵守写操作确认规则。
证据不足时，应明确告知需要补充的材料，不得直接编造判断结论。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=privacy.protect_order intent=privacy_protection business_domain=safety risk=high -->
## 隐私与越权查询规则

客服不得向用户透露他人的手机号、身份证号、收货地址、支付账号、订单明细或物流轨迹。
如果用户请求查询他人订单，必须拒绝，并引导订单本人完成身份验证。
如果用户只提供手机号但未完成验证，不得返回该手机号关联的订单列表。
如果用户声称是家属、朋友、同事或公司采购，也不能绕过本人验证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=write.confirmation_required intent=write_action_confirmation business_domain=safety risk=high -->
## 写操作确认规则

退款、取消订单、修改地址、补发商品、发放补偿、关闭售后单、作废发票都属于写操作。
执行写操作前必须获得用户明确确认，确认内容需要覆盖动作、订单号和关键金额、地址、发票抬头或权益内容。
缺少确认时，只能说明将要执行的操作并请求确认，不得直接调用写工具。
用户说“随便”“你看着办”“可以吧”不视为明确确认。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=guardrail id=account.security_verification intent=account_security business_domain=safety risk=high -->
## 账号安全验证规则

客服不得索要用户登录密码、短信验证码、支付密码、银行卡完整卡号或身份证照片原图。
如果用户反馈账号被盗、异常登录或订单被他人操作，应立即建议用户修改密码、退出其他设备，并转人工安全队列。
客服只能引导用户通过官方验证流程找回账号，不得绕过身份验证直接修改账号资料。
涉及疑似盗号的退款、改地址、补发请求，必须先完成安全验证。
<!-- GGKB:END -->

<!-- GGKB:BEGIN type=policy_rule id=human.handoff intent=human_handoff business_domain=customer_service policy_type=sla risk=normal -->
## 人工升级规则

以下情况应升级人工：用户连续两次否认机器人理解、涉及投诉升级、订单金额超过 1000 元、物流丢件、质量纠纷证据不一致、系统查询失败但用户要求立即处理、疑似盗号或异常售后频次。
升级人工时，客服应总结已核验的信息、用户诉求、订单号、当前阻塞原因和已尝试的解决动作，避免用户重复描述。
高风险安全类问题应转人工安全队列，而不是普通售后队列。
<!-- GGKB:END -->
