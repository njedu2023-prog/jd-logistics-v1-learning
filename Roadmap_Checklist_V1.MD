# Roadmap / Checklist V1

## Milestone 1：文档收口
- [ ] 扩写《单票标签定义规范 V1》
- [ ] 扩写《单票特征体系规范 V1》
- [ ] 扩写《单票滚动训练与验收规范 V1》

## Milestone 2：数据底座（JD-data）
- [ ] 历史数据追加写入（不覆盖 latest）
- [ ] 交易日历对齐 + quality_flag
- [ ] data_version + refresh_log 落地

## Milestone 3：工程化（主仓）
- [ ] 规范目录：data_raw/data_clean/data_base/data_context/data_events
- [ ] features/labels 生成脚本
- [ ] models/predictions/evaluation 输出一致口径

## Milestone 4：训练与验收
- [ ] walk-forward + 4 horizon（T+1/5/20/120）
- [ ] quantile 输出 + 覆盖率监控

说明：每个任务完成后记录版本与变更说明。
