# -*- coding: utf-8 -*-
"""
京东物流 V1.0 —— 完整预测执行器（GitHub闭环版）
流程：
1) 读取 state/model_state.json
2) 拉取唯一主源的“收盘后真实数据”（方案A阻塞式重试）
3) 写入 market/market_daily.jsonl
4) 生成预测 -> 写入 predictions/pred_daily.jsonl
5) 评估昨日预测 -> 写入 evaluation/eval_daily.jsonl（拿不到则写 null + 原因）
6) 自学习：根据评估与昨日/今日数据更新 mu/sigma（有数据才更新）-> 写入 learning/param_updates.jsonl
7) 覆盖写 state/model_state.json
8) 输出“全模块报告”到 stdout，并写入 reports/latest_report.md（便于留痕）
"""

import os
import json
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List, Tuple

# ---------- 常量 ----------
MODEL_VERSION = "V1.0"
SYMBOL = os.getenv("SYMBOL", "02618.HK")

# 唯一主源URL（你可在 Actions env 改）
DATA_SOURCE_URL = os.getenv(
    "DATA_SOURCE_URL",
    "https://raw.githubusercontent.com/njedu2023-prog/xiaomi-data/main/jd-logistics-latest.json"
)

# 方案A：阻塞式重试（默认 0s, 300s, 300s）
RETRY_DELAYS = [0, 300, 300]

# 文件路径
RUN_LOG_PATH = "runs/run_log.jsonl"
MARKET_PATH = "market/market_daily.jsonl"
PRED_PATH = "predictions/pred_daily.jsonl"
EVAL_PATH = "evaluation/eval_daily.jsonl"
UPD_PATH = "learning/param_updates.jsonl"
STATE_PATH = "state/model_state.json"
REPORT_PATH = "reports/latest_report.md"

# 正态分位 z 值（5/25/50/75/95）
Z = {
    0.05: -1.6448536269514722,
    0.25: -0.6744897501960817,
    0.50:  0.0,
    0.75:  0.6744897501960817,
    0.95:  1.6448536269514722,
}

# 交易日年化：252
TRADING_DAYS = 252.0

# ---------- 工具 ----------
def now_bjt_iso() -> str:
    # 北京时间 UTC+8
    bjt = timezone(timedelta(hours=8))
    return datetime.now(tz=bjt).isoformat(timespec="seconds")

def bjt_date_str() -> str:
    bjt = timezone(timedelta(hours=8))
    return datetime.now(tz=bjt).strftime("%Y-%m-%d")

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir_for_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def clamp(x: float, lo: float, hi: float) -> Tuple[float, bool]:
    if x < lo:
        return lo, True
    if x > hi:
        return hi, True
    return x, False

def gbm_quantile(s0: float, mu: float, sigma: float, t_years: float, z: float) -> float:
    # S = S0 * exp((mu - 0.5*sigma^2)*t + sigma*sqrt(t)*z)
    return s0 * math.exp((mu - 0.5 * sigma * sigma) * t_years + sigma * math.sqrt(t_years) * z)

def calc_hit_band(actual: float, p05: float, p25: float, p50: float, p75: float, p95: float) -> str:
    if actual <= p05:
        return "<=P05"
    if actual <= p25:
        return "P05-P25"
    if actual <= p50:
        return "P25-P50"
    if actual <= p75:
        return "P50-P75"
    if actual <= p95:
        return "P75-P95"
    return ">=P95"

# ---------- 主源数据抓取 ----------
def fetch_market_json_with_retry(url: str) -> Tuple[Optional[Dict[str, Any]], int, Optional[str]]:
    import requests  # requirements.txt

    last_err = None
    retries = 0

    for i, d in enumerate(RETRY_DELAYS):
        if d > 0:
            time.sleep(d)
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            data = r.json()
            return data, retries, None
        except Exception as e:
            last_err = repr(e)
            if i < len(RETRY_DELAYS) - 1:
                retries += 1
            continue

    return None, retries, last_err

def parse_market_payload(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    你主源 jd-logistics-latest.json 字段未知时：只做“兼容映射”，拿不到就返回原因（不编）。
    期望最少：trade_date + close + volume + amount_hkd（或 amount）
    """
    def pick(*keys):
        for k in keys:
            if k in payload:
                return payload[k]
        return None

    trade_date = pick("trade_date", "date", "tradeDate", "日期")
    open_ = pick("open", "open_price", "开盘", "Open")
    high = pick("high", "high_price", "最高", "High")
    low  = pick("low", "low_price", "最低", "Low")
    close = pick("close", "close_price", "收盘", "Close")
    volume = pick("volume", "vol", "成交量", "Volume")
    amount = pick("amount_hkd", "amount", "成交额", "Turnover")
    source_ts = pick("source_timestamp_bjt", "timestamp_bjt", "timestamp", "数据时间")

    # 基础校验
    if trade_date is None:
        return None, "主源缺少 trade_date/date 字段"
    if close is None:
        return None, "主源缺少 close/收盘 字段"
    if volume is None:
        return None, "主源缺少 volume/成交量 字段"
    if amount is None:
        return None, "主源缺少 amount/成交额 字段"

    try:
        close_f = float(close)
        vol_i = int(float(volume))
        amt_i = int(float(amount))
    except Exception:
        return None, "主源字段类型无法解析为数值（close/volume/amount）"

    # open/high/low 允许缺失，但若存在则解析
    def to_float_or_none(x):
        if x is None or x == "":
            return None
        try:
            return float(x)
        except Exception:
            return None

    open_f = to_float_or_none(open_)
    high_f = to_float_or_none(high)
    low_f  = to_float_or_none(low)

    # integrity：只做不编的逻辑自检
    integrity_ok = True
    integrity_note = None
    if open_f is not None and high_f is not None and low_f is not None:
        if not (low_f <= open_f <= high_f):
            integrity_ok = False
            integrity_note = "OHLC自检失败：open不在[low,high]内"
    if high_f is not None and low_f is not None:
        if not (low_f <= close_f <= high_f):
            integrity_ok = False
            integrity_note = "OHLC自检失败：close不在[low,high]内"
    if close_f <= 0:
        integrity_ok = False
        integrity_note = "收盘价<=0"

    amount_yi = amt_i / 1e8

    record = {
        "trade_date": str(trade_date),
        "symbol": SYMBOL,
        "open": open_f if open_f is not None else None,
        "high": high_f if high_f is not None else None,
        "low":  low_f if low_f is not None else None,
        "close": close_f,
        "volume": vol_i,
        "amount_hkd": amt_i,
        "amount_yi_hkd": amount_yi,
        "source_url": DATA_SOURCE_URL,
        "source_timestamp_bjt": str(source_ts) if source_ts is not None else None,
        "integrity_ok": bool(integrity_ok),
        "integrity_note": integrity_note,
        "model_version": MODEL_VERSION,  # 你文档虽然没要求 market_daily 带，但加不影响；如你要严格不加可删
    }
    return record, None

# ---------- 预测 / 评估 / 学习 ----------
def load_or_init_state() -> Dict[str, Any]:
    st = load_json(STATE_PATH)
    if st is not None:
        return st

    # 默认初始化（可按你之前冻结参数改）
    default = {
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "last_success_trade_date": "1970-01-01",
        "last_run_id": "",
        "mu_base": 0.0,
        "sigma_short": 0.30,
        "sigma_mid": 0.35,
        "sigma_long": 0.40,
        "enable_volume_factor": True,
        "enable_beta_anchor": True,
        "beta_window": 60,
        "volume_window": 20,
        "safety_limits": {
            "mu_min": -1.0,
            "mu_max":  1.0,
            "sigma_short_min": 0.05,
            "sigma_short_max": 1.50,
            "sigma_mid_min": 0.05,
            "sigma_mid_max": 1.50,
            "sigma_long_min": 0.05,
            "sigma_long_max": 1.50,
        },
        "updated_at_bjt": now_bjt_iso()
    }
    return default

def find_prev_market_close(trade_date: str) -> Optional[float]:
    rows = read_jsonl(MARKET_PATH)
    # 找到严格小于 trade_date 的最近一条
    candidates = [r for r in rows if r.get("trade_date") and r["trade_date"] < trade_date and r.get("close") is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["trade_date"])
    return float(candidates[-1]["close"])

def find_yesterday_pred_for_today(today_trade_date: str) -> Optional[Dict[str, Any]]:
    preds = read_jsonl(PRED_PATH)
    # yesterday 预测里：target_trade_date_t1 == today_trade_date
    cands = [p for p in preds if p.get("target_trade_date_t1") == today_trade_date]
    if not cands:
        return None
    cands.sort(key=lambda x: x.get("pred_date", ""))
    return cands[-1]

def make_predictions(state: Dict[str, Any], today_close: float, today_trade_date: str) -> Dict[str, Any]:
    mu_raw = state.get("mu_base")

    # 冷启动兜底值（只在第一次）
    if mu_raw is None:
        mu_raw = 0.0

    def build_prediction_output(
        mu_raw: float,
        state: Dict[str, Any],
        today_trade_date: str,
        today_close: float,
    ) -> Dict[str, Any]:
        mu = float(mu_raw)
        sig_s = float(state["sigma_short"])
        sig_m = float(state["sigma_mid"])
        sig_l = float(state["sigma_long"])

        # 时间尺度（年）
        t1 = 1.0 / TRADING_DAYS
        m1 = 21.0 / TRADING_DAYS
        m6 = 126.0 / TRADING_DAYS

        out = {
            "pred_date": today_trade_date,  # 收盘后生成
            "target_trade_date_t1": today_trade_date,  # 若有交易日历可替换为 T+1
            "symbol": SYMBOL,

            "module3_t1_p05": gbm_quantile(today_close, mu, sig_s, t1, Z[0.05]),
            "module3_t1_p25": gbm_quantile(today_close, mu, sig_s, t1, Z[0.25]),
            "module3_t1_p50": gbm_quantile(today_close, mu, sig_s, t1, Z[0.50]),
            "module3_t1_p75": gbm_quantile(today_close, mu, sig_s, t1, Z[0.75]),
            "module3_t1_p95": gbm_quantile(today_close, mu, sig_s, t1, Z[0.95]),

            "module4_m1_p05": gbm_quantile(today_close, mu, sig_m, m1, Z[0.05]),
            "module4_m1_p25": gbm_quantile(today_close, mu, sig_m, m1, Z[0.25]),
            "module4_m1_p50": gbm_quantile(today_close, mu, sig_m, m1, Z[0.50]),
            "module4_m1_p75": gbm_quantile(today_close, mu, sig_m, m1, Z[0.75]),
            "module4_m1_p95": gbm_quantile(today_close, mu, sig_m, m1, Z[0.95]),

            "module5_m6_p05": gbm_quantile(today_close, mu, sig_l, m6, Z[0.05]),
            "module5_m6_p25": gbm_quantile(today_close, mu, sig_l, m6, Z[0.25]),
            "module5_m6_p50": gbm_quantile(today_close, mu, sig_l, m6, Z[0.50]),
            "module5_m6_p75": gbm_quantile(today_close, mu, sig_l, m6, Z[0.75]),
            "module5_m6_p95": gbm_quantile(today_close, mu, sig_l, m6, Z[0.95]),

            "mu_base": mu,
            "sigma_short": sig_s,
            "sigma_mid": sig_m,
            "sigma_long": sig_l,
            "enable_volume_factor": bool(state["enable_volume_factor"]),
            "enable_beta_anchor": bool(state["enable_beta_anchor"]),
            "run_id": state.get("last_run_id", ""),
            "model_version": MODEL_VERSION,
        }

        return out
    
    def maybe_self_learn_and_update_state(
    state: Dict[str, Any],
    today_trade_date: str,
    today_close: float,
    eval_row: Optional[Dict[str, Any]],
    run_id: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    简化自学习：仅在具备“昨日close + 今日close + 评估记录”时更新：
    - mu_base：向“真实日收益年化”做 EWMA 微调
    - sigma_short：根据相对误差做 EWMA 微调
    任何缺数据 -> 不更新参数，不写 param_updates
    """
    updates = []
    safety = state["safety_limits"]

    prev_close = find_prev_market_close(today_trade_date)
    if prev_close is None:
        return state, updates
    if eval_row is None or eval_row.get("rel_error_pct") is None:
        return state, updates

    # 真实日对数收益（年化）
    try:
        r = math.log(today_close / float(prev_close))
    except Exception:
        return state, updates

    realized_mu_annual = r * TRADING_DAYS
    old_mu = float(state["mu_base"])
    new_mu_raw = 0.90 * old_mu + 0.10 * realized_mu_annual

    new_mu, clamped_mu = clamp(new_mu_raw, float(safety["mu_min"]), float(safety["mu_max"]))
    if new_mu != old_mu:
        updates.append({
            "update_id": str(uuid.uuid4()),
            "update_date": today_trade_date,
            "symbol": SYMBOL,
            "param_name": "mu_base",
            "old_value": old_mu,
            "new_value": new_mu,
            "delta": new_mu - old_mu,
            "trigger_metric": "rel_error_pct",
            "trigger_window": "last_1_trading_day",
            "trigger_rule": "EWMA(0.9*old + 0.1*realized_mu_annual)，并受 safety_limits 钳制",
            "evidence_ref": f"{eval_row.get('eval_date')}|{eval_row.get('target_trade_date')}",
            "safety_clamp_applied": bool(clamped_mu),
            "model_version": MODEL_VERSION
        })

    # sigma_short 用相对误差作微调（示例规则：误差越大，sigma 轻微上调；误差很小则轻微下调）
    rel_err = float(eval_row["rel_error_pct"])
    old_sig = float(state["sigma_short"])
    target_sig = max(0.05, min(1.5, old_sig * (1.0 + (rel_err - 0.01))))  # 0.01 视为“正常”基准
    new_sig_raw = 0.90 * old_sig + 0.10 * target_sig

    new_sig, clamped_sig = clamp(new_sig_raw, float(safety["sigma_short_min"]), float(safety["sigma_short_max"]))
    if new_sig != old_sig:
        updates.append({
            "update_id": str(uuid.uuid4()),
            "update_date": today_trade_date,
            "symbol": SYMBOL,
            "param_name": "sigma_short",
            "old_value": old_sig,
            "new_value": new_sig,
            "delta": new_sig - old_sig,
            "trigger_metric": "rel_error_pct",
            "trigger_window": "last_1_trading_day",
            "trigger_rule": "EWMA(0.9*old + 0.1*target_sigma_from_error)，并受 safety_limits 钳制",
            "evidence_ref": f"{eval_row.get('eval_date')}|{eval_row.get('target_trade_date')}",
            "safety_clamp_applied": bool(clamped_sig),
            "model_version": MODEL_VERSION
        })

    # 应用更新
    if updates:
        state["mu_base"] = new_mu
        state["sigma_short"] = new_sig

    return state, updates

# ---------- 报告 ----------
def fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "null"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def build_report(
    market_row: Optional[Dict[str, Any]],
    eval_row: Optional[Dict[str, Any]],
    pred_row: Optional[Dict[str, Any]],
    state: Dict[str, Any],
    run_log: Dict[str, Any],
    param_updates: List[Dict[str, Any]]
) -> str:
    lines = []
    lines.append(f"# 京东物流 V1.0 全模块报告（{bjt_date_str()}）")
    lines.append("")
    lines.append(f"- run_id: {run_log.get('run_id')}")
    lines.append(f"- trigger_time_bjt: {run_log.get('trigger_time_bjt')}")
    lines.append(f"- data_source_url: {run_log.get('data_source_url')}")
    lines.append(f"- fetch_status: {run_log.get('fetch_status')} (retries={run_log.get('fetch_retries')})")
    lines.append("")

    # ① 当日实际交易数据
    lines.append("## ① 当日实际交易数据")
    if market_row is None:
        lines.append("- 数据不可用（已按方案A重试）")
    else:
        lines.append(f"- 交易日：{market_row.get('trade_date')}")
        lines.append(f"- 开盘：{fmt(market_row.get('open'))} HKD")
        lines.append(f"- 最高：{fmt(market_row.get('high'))} HKD")
        lines.append(f"- 最低：{fmt(market_row.get('low'))} HKD")
        lines.append(f"- 收盘：{fmt(market_row.get('close'))} HKD")
        lines.append(f"- 成交量：{fmt(market_row.get('volume'))} 股")
        lines.append(f"- 成交额：{fmt(market_row.get('amount_yi_hkd'))}（亿港元）")
        lines.append(f"- 数据来源：{market_row.get('source_url')}")
        lines.append(f"- 自检：integrity_ok={market_row.get('integrity_ok')} note={market_row.get('integrity_note')}")
    lines.append("")

    # ② 昨日预测回顾
    lines.append("## ② 昨日预测回顾")
    if eval_row is None:
        lines.append("- 无法评估：缺少“昨日对今日的预测记录”或字段不足（按规则不编，已写入 eval_daily 为 null/跳过）")
    else:
        lines.append(f"- 今日实际收盘：{fmt(eval_row.get('actual_close'))}")
        lines.append(f"- 昨日T+1中位预测：{fmt(eval_row.get('pred_median_t1'))}")
        lines.append(f"- 误差：abs={fmt(eval_row.get('abs_error'))}, rel={fmt(eval_row.get('rel_error_pct'), 4)}")
        lines.append(f"- 命中区间：{eval_row.get('hit_band')}")
    lines.append("")

    # ③ 次日价格分布预测（T+1）
    lines.append("## ③ 次日价格分布预测（T+1）")
    if pred_row is None:
        lines.append("- 无预测（因当日真实数据不可用）")
    else:
        lines.append(f"- P05 {fmt(pred_row.get('module3_t1_p05'))} | P25 {fmt(pred_row.get('module3_t1_p25'))} | P50 {fmt(pred_row.get('module3_t1_p50'))} | P75 {fmt(pred_row.get('module3_t1_p75'))} | P95 {fmt(pred_row.get('module3_t1_p95'))}")
    lines.append("")

    # ④ 未来1个月价格分布预测
    lines.append("## ④ 未来1个月价格分布预测")
    if pred_row is None:
        lines.append("- 无预测（因当日真实数据不可用）")
    else:
        lines.append(f"- P05 {fmt(pred_row.get('module4_m1_p05'))} | P25 {fmt(pred_row.get('module4_m1_p25'))} | P50 {fmt(pred_row.get('module4_m1_p50'))} | P75 {fmt(pred_row.get('module4_m1_p75'))} | P95 {fmt(pred_row.get('module4_m1_p95'))}")
    lines.append("")

    # ⑤ 未来6个月价格分布预测
    lines.append("## ⑤ 未来6个月价格分布预测")
    if pred_row is None:
        lines.append("- 无预测（因当日真实数据不可用）")
    else:
        lines.append(f"- P05 {fmt(pred_row.get('module5_m6_p05'))} | P25 {fmt(pred_row.get('module5_m6_p25'))} | P50 {fmt(pred_row.get('module5_m6_p50'))} | P75 {fmt(pred_row.get('module5_m6_p75'))} | P95 {fmt(pred_row.get('module5_m6_p95'))}")
    lines.append("")

    # ⑥ 模型状态与学习更新
    lines.append("## ⑥ 模型状态与学习更新")
    lines.append(f"- mu_base: {fmt(state.get('mu_base'), 6)}")
    lines.append(f"- sigma_short/mid/long: {fmt(state.get('sigma_short'), 6)} / {fmt(state.get('sigma_mid'), 6)} / {fmt(state.get('sigma_long'), 6)}")
    lines.append(f"- enable_volume_factor: {state.get('enable_volume_factor')}, enable_beta_anchor: {state.get('enable_beta_anchor')}")
    lines.append(f"- last_success_trade_date: {state.get('last_success_trade_date')}")
    lines.append(f"- updated_at_bjt: {state.get('updated_at_bjt')}")
    if not param_updates:
        lines.append("- 本次无参数更新（数据不足或未触发规则）")
    else:
        for u in param_updates:
            lines.append(f"- 更新 {u['param_name']}: {fmt(u['old_value'],6)} -> {fmt(u['new_value'],6)} (Δ={fmt(u['delta'],6)}) clamp={u['safety_clamp_applied']}")
    lines.append("")

    return "\n".join(lines)

# ---------- 主程序 ----------
def main():
    run_id = str(uuid.uuid4())
    trigger_time = now_bjt_iso()
    target_trade_date = bjt_date_str()

    run_log = {
        "run_id": run_id,
        "run_type": os.getenv("RUN_TYPE", "AUTO"),
        "trigger_time_bjt": trigger_time,
        "target_trade_date": target_trade_date,
        "data_source_url": DATA_SOURCE_URL,
        "fetch_status": "FAILED",
        "fetch_retries": 0,
        "fetch_error": None,
        "report_output_status": "NO_OUTPUT",
        "report_reason": "",
        "cache_write_status": "SKIPPED",
        "cache_error": None,
        "model_version": MODEL_VERSION
    }

    state = load_or_init_state()
    state["last_run_id"] = run_id

    payload, retries, fetch_err = fetch_market_json_with_retry(DATA_SOURCE_URL)
    run_log["fetch_retries"] = retries

    market_row = None
    pred_row = None
    eval_row = None
    param_updates = []

    if payload is None:
        run_log["fetch_status"] = "FAILED"
        run_log["fetch_error"] = fetch_err
        run_log["report_reason"] = f"主源拉取失败（已按方案A重试）：{fetch_err}"
        run_log["report_output_status"] = "NO_OUTPUT"
        # 写 run_log（即使失败也要留痕）
        append_jsonl(RUN_LOG_PATH, run_log)
        report = build_report(None, None, None, state, run_log, [])
        ensure_dir_for_file(REPORT_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

    market_row, parse_err = parse_market_payload(payload)
    if market_row is None:
        run_log["fetch_status"] = "FAILED"
        run_log["fetch_error"] = parse_err
        run_log["report_reason"] = f"主源数据解析失败：{parse_err}"
        run_log["report_output_status"] = "NO_OUTPUT"
        append_jsonl(RUN_LOG_PATH, run_log)
        report = build_report(None, None, None, state, run_log, [])
        ensure_dir_for_file(REPORT_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

    # 用“主源给的交易日”为准，不用本地日期硬推
    target_trade_date = market_row["trade_date"]
    run_log["target_trade_date"] = target_trade_date
    run_log["fetch_status"] = "SUCCESS"
    run_log["fetch_error"] = None

    # 写 market_daily
    try:
        append_jsonl(MARKET_PATH, {k: market_row.get(k) for k in [
            "trade_date","symbol","open","high","low","close","volume","amount_hkd","amount_yi_hkd",
            "source_url","source_timestamp_bjt","integrity_ok","integrity_note"
        ]})
        run_log["cache_write_status"] = "WROTE"
        run_log["cache_error"] = None
        state["last_success_trade_date"] = target_trade_date
    except Exception as e:
        run_log["cache_write_status"] = "FAILED"
        run_log["cache_error"] = repr(e)

    # 评估昨日预测（若存在）
    ypred = find_yesterday_pred_for_today(target_trade_date)
    if ypred is not None:
        actual_close = float(market_row["close"])
        pred_median = ypred.get("module3_t1_p50")
        pred_p25 = ypred.get("module3_t1_p25")
        pred_p75 = ypred.get("module3_t1_p75")

        # abs_error 必填，但如果昨日预测字段缺失，不编：写 null 并说明（这里选择：不写 eval_daily，避免违背必填；你若要强制写，可改成写 null）
        if pred_median is not None and pred_p25 is not None and pred_p75 is not None:
            abs_error = abs(float(pred_median) - actual_close)
            rel_error = abs_error / actual_close if actual_close != 0 else None
            hit_band = calc_hit_band(
                actual_close,
                float(ypred.get("module3_t1_p05")),
                float(ypred.get("module3_t1_p25")),
                float(ypred.get("module3_t1_p50")),
                float(ypred.get("module3_t1_p75")),
                float(ypred.get("module3_t1_p95")),
            )

            eval_row = {
                "eval_date": target_trade_date,
                "target_trade_date": target_trade_date,
                "symbol": SYMBOL,
                "actual_close": actual_close,
                "pred_ref_date": ypred.get("pred_date"),
                "pred_median_t1": float(pred_median),
                "pred_p25_t1": float(pred_p25),
                "pred_p75_t1": float(pred_p75),
                "abs_error": abs_error,
                "rel_error_pct": rel_error,
                "hit_band": hit_band,
                "amount_yi_hkd": float(market_row["amount_yi_hkd"]),
                "volume_percentile_20d": None,  # 拿不到就 null（合规）
                "enable_volume_factor": bool(ypred.get("enable_volume_factor", False)),
                "enable_beta_anchor": bool(ypred.get("enable_beta_anchor", False)),
                "run_id": run_id,
                "model_version": MODEL_VERSION
            }
            append_jsonl(EVAL_PATH, eval_row)

    # 自学习更新（有足够数据才更新）
    state, param_updates = maybe_self_learn_and_update_state(
        state=state,
        today_trade_date=target_trade_date,
        today_close=float(market_row["close"]),
        eval_row=eval_row,
        run_id=run_id
    )
    for u in param_updates:
        append_jsonl(UPD_PATH, u)

    # 生成预测并写入 pred_daily
    pred_row = make_predictions(state, float(market_row["close"]), target_trade_date)
    pred_row["run_id"] = run_id
    append_jsonl(PRED_PATH, pred_row)

    # 覆盖写 state
    state["updated_at_bjt"] = now_bjt_iso()
    save_json(STATE_PATH, state)

    run_log["report_output_status"] = "OUTPUT"
    run_log["report_reason"] = "当日真实数据拉取成功；预测/写入完成。"
    append_jsonl(RUN_LOG_PATH, run_log)

    report = build_report(market_row, eval_row, pred_row, state, run_log, param_updates)
    ensure_dir_for_file(REPORT_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)

if __name__ == "__main__":
    main()
