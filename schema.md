# 京东物流 V1.0 自我学习数据仓库（GitHub 方案）

## 1. 模型标识
- model_name: 京东物流 V1.0
- model_version: V1.0
- timezone: Asia/Shanghai（北京时间）

## 2. 地址层冻结（URL 不得变更，只更新文件内容）
- run_log.jsonl: runs/run_log.jsonl
- market_daily.jsonl: market/market_daily.jsonl
- pred_daily.jsonl: predictions/pred_daily.jsonl
- eval_daily.jsonl: evaluation/eval_daily.jsonl
- param_updates.jsonl: learning/param_updates.jsonl
- model_state.json: state/model_state.json

## 3. 合规规则（系统级冻结）
- 严禁估算 / 回填 / 推理补数
- 若真实数据不可得：写 null 并记录原因（不得编造）
- 模块①唯一主源：GitHub Raw JSON（jd-logistics-latest.json）
- 方案A：有限次阻塞式重试（抓不到数据则重试，直至成功或达上限）
- 成交额展示口径（报告模块①冻结）：成交额：X（亿港元）

## 4. 文件格式
- *.jsonl：JSON Lines，一行一个 JSON 对象（真实记录）。建议初始为空文件。
- model_state.json：最新状态快照，每次运行覆盖更新。
