# Schema  

本仓库用于存储《京东物流 V1.0》自我学习模块所需的各类数据文件。  

目录结构：  

- runs/run_log.jsonl：记录每次执行日志。  
- market/market_daily.jsonl：存储当日真实行情数据。  
- predictions/pred_daily.jsonl：存储每日预测结果。  
- evaluation/eval_daily.jsonl：记录昨日预测与今日实际的对比。  
- learning/param_updates.jsonl：记录每次参数更新详情。  
- state/model_state.json：存储当前模型状态快照，包括 safety_limits 等配置。  

详细字段定义请参考系统文档。
