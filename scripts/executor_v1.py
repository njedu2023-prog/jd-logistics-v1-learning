# -*- coding: utf-8 -*-
"""
JD Logistics V1.0 - Executor (runnable baseline)
- 目的：先把 GitHub Actions 执行链路跑通，修复缩进/函数块错误
- 数据源：GitHub Raw JSON（模块①当日真实交易数据）
- 输出：run_log / eval / updates / pred_daily / state / report
"""

import json
import os
import sys
import time
import uuid
import math
import hashlib
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None


# =========================
# 固定配置（按你当前冻结规则）
# =========================
MODEL_VERSION = "V1.0"
SYMBOL = "02618.HK"

# 模块①唯一数据源（你已锁定）
MARKET_URL = "https://raw.githubusercontent.com/njedu2023-prog/xiaomi-data/main/jd-logistics-latest.json"

# 输出路径（仓库内）
RUN_LOG_PATH = "runs/run_log.jsonl"
EVAL_PATH = "evaluation/eval.jsonl"
UPD_PATH = "learning/updates.jsonl"
PRED_PATH = "predictions/pred_daily.jsonl"
STATE_PATH = "state/model_state.json"
REPORT_PATH = "report_latest.md"

# 抓取策略：方案A（有限次阻塞重试）
RETRY_MAX = 12           # 最多重试次数
RETRY_SLEEP_SEC = 10     # 每次间隔秒数
HTTP_TIMEOUT = 15


# =========================
# 基础工具
# =========================
def now_bjt() -> dt.datetime:
    return dt.datetime.utcnow() + dt.timedelta(hours=8)

def now_bjt_iso() -> str:
    return now_bjt().replace(microsecond=0).isoformat()

def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir_for_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def sha1_of_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default

def pct(a: float, b: float) -> float:
    # 误差百分比 abs(a-b)/b *100，b=0 时避免除零
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100.0

def fetch_module_1_market():
    """
    模块①：当日真实交易数据（唯一数据源）
    """
    if requests is None:
        raise RuntimeError("requests not available")

    last_err = None
    for _ in range(RETRY_MAX):
        try:
            resp = requests.get(MARKET_URL, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            # 上游字段适配
            trade_date = str(data.get("date", "")).strip()
            amount = data.get("amount", None)

            if not trade_date or amount is None:
                raise ValueError("missing trade_date or amount")

            amount_yi_hkd = round(float(amount) / 1e8, 4)

            return {
                "symbol": SYMBOL,
                "trade_date": trade_date,
                "open": data.get("open", ""),
                "high": data.get("high", ""),
                "low": data.get("low", ""),
                "close": data.get("close", ""),
                "volume": data.get("volume", ""),
                "amount_yi_hkd": amount_yi_hkd,
            }

        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP_SEC)

    raise RuntimeError(f"module_1_market failed: {last_err}")

# =========================
# 数据抓取（方案A阻塞重试）
# =========================
def http_get_text(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests 未安装。请在 requirements.txt 里加入 requests。")
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

def fetch_market_row_blocking(url: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    方案A：有限次阻塞重试；成功返回 market_row；失败返回 None
    """
    diag = {
        "source_url": url,
        "attempts": 0,
        "ok": False,
        "last_error": None,
    }

    for i in range(1, RETRY_MAX + 1):
        diag["attempts"] = i
        try:
            txt = http_get_text(url)
            raw = json.loads(txt)
            if not isinstance(raw, dict):
                raise ValueError("JSON不是对象(dict)")

            # -------- 兼容字段映射（不推断，只做确定性重命名/换算） --------
            market = dict(raw)

            # 情况A：date -> trade_date
            if "trade_date" not in market and "date" in market:
                market["trade_date"] = market.get("date")

            # 情况A：amount(港元) -> amount_yi_hkd(亿港元)
            if "amount_yi_hkd" not in market and "amount" in market:
                amt = market.get("amount", None)
                if amt is None:
                    raise ValueError("amount为空")
                market["amount_yi_hkd"] = round(float(amt) / 1e8, 4)

            # -------- 基本字段校验（不推断、不回填） --------
            required = ["trade_date", "open", "high", "low", "close", "volume", "amount_yi_hkd"]
            missing = [k for k in required if k not in market or market[k] in ("", None)]
            if missing:
                raise ValueError(f"JSON缺字段或为空: {missing}")

            # -------- 强制类型 --------
            market_row = {
                "trade_date": str(market["trade_date"]),
                "open": float(market["open"]),
                "high": float(market["high"]),
                "low": float(market["low"]),
                "close": float(market["close"]),
                "volume": int(float(market["volume"])),
                "amount_yi_hkd": float(market["amount_yi_hkd"]),
                "source": "github_raw_json",
                "source_url": url,
                "fetched_at_bjt": now_bjt_iso(),
                "raw_sha1": sha1_of_text(txt),
            }

            diag["ok"] = True
            return market_row, diag

        except Exception as e:
            diag["last_error"] = str(e)
            time.sleep(RETRY_SLEEP_SEC)

    return None, diag



# =========================
# 状态/学习（最小可跑通版）
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "created_at_bjt": now_bjt_iso(),
        "updated_at_bjt": now_bjt_iso(),
        # 简化参数：波动估计（可被自学习更新）
        "sigma_base": 0.035,     # 3.5% 作为初始波动
        "mu_base": 0.0,          # 漂移先置 0（合规保守）
        # 记录最近一次收盘（用于回顾）
        "last_close": None,
        "last_trade_date": None,
    }

def load_state() -> Dict[str, Any]:
    st = read_json(STATE_PATH, default=None)
    if not isinstance(st, dict):
        st = default_state()
        save_json(STATE_PATH, st)
    # 兼容缺字段
    for k, v in default_state().items():
        if k not in st:
            st[k] = v
    return st

def maybe_self_learn_and_update_state(
    state: Dict[str, Any],
    today_trade_date: str,
    today_close: float,
    eval_row: Dict[str, Any],
    run_id: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    最小学习：用今日相对误差来轻微调整 sigma_base（不做激进更新）
    有 eval_row 才更新；否则不更新
    """
    updates: List[Dict[str, Any]] = []
    try:
        rel_err = safe_float(eval_row.get("rel_error_pct"), None)
        if rel_err is None:
            return state, updates

        old_sigma = float(state.get("sigma_base", 0.035))
        # 用误差对 sigma 做一个很小的 EMA 修正（仅保底）
        # 把误差%转成比例
        err = max(0.0, rel_err / 100.0)
        new_sigma = 0.9 * old_sigma + 0.1 * max(0.01, min(0.20, err))
        state["sigma_base"] = float(new_sigma)

        upd = {
            "update_time_bjt": now_bjt_iso(),
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "type": "self_learn_sigma",
            "today_trade_date": today_trade_date,
            "old_sigma_base": old_sigma,
            "new_sigma_base": float(new_sigma),
            "note": "最小自学习：用今日预测误差对sigma做轻微EMA校准（保底实现）"
        }
        updates.append(upd)
    except Exception:
        # 学习失败不影响主流程
        pass

    return state, updates


# =========================
# 预测（最小可跑通版）
# =========================
def normal_quantile(p: float) -> float:
    """
    近似标准正态分位数（无需 scipy）
    使用 Peter J. Acklam 近似（简化实现）
    """
    # 边界
    p = min(max(p, 1e-10), 1 - 1e-10)
    # 系数（Acklam）
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def make_predictions(state: Dict[str, Any], spot_close: float, target_trade_date: str) -> Dict[str, Any]:
    """
    生成分位点预测：T+1 / 1个月 / 6个月（最小可跑通版）
    """
    sigma = float(state.get("sigma_base", 0.035))
    mu_raw = state.get("mu_base", 0.0)
    mu = float(0.0 if mu_raw is None else mu_raw)

    sigma_raw = state.get("sigma_base", 0.035)
    sigma = float(0.035 if sigma_raw is None else sigma_raw)



    # 时间尺度（近似交易日）
    t1 = 1/252
    t1m = 21/252
    t6m = 126/252

    def q_price(t: float, p: float) -> float:
        z = normal_quantile(p)
        # 简化：对数正态
        return float(spot_close * math.exp((mu - 0.5*sigma*sigma)*t + sigma*math.sqrt(t)*z))

    pred = {
        "pred_date": target_trade_date,
        "symbol": SYMBOL,
        "model_version": MODEL_VERSION,
        "sigma_base": sigma,
        "mu_base": mu,
        # 开关（保持字段以兼容你之前结构）
        "enable_volume_factor": False,
        "enable_beta_anchor": False,

        # T+1
        "module3_t1_p05": q_price(t1, 0.05),
        "module3_t1_p25": q_price(t1, 0.25),
        "module3_t1_p50": q_price(t1, 0.50),
        "module3_t1_p75": q_price(t1, 0.75),
        "module3_t1_p95": q_price(t1, 0.95),

        # 1个月
        "module4_1m_p05": q_price(t1m, 0.05),
        "module4_1m_p25": q_price(t1m, 0.25),
        "module4_1m_p50": q_price(t1m, 0.50),
        "module4_1m_p75": q_price(t1m, 0.75),
        "module4_1m_p95": q_price(t1m, 0.95),

        # 6个月
        "module5_6m_p05": q_price(t6m, 0.05),
        "module5_6m_p25": q_price(t6m, 0.25),
        "module5_6m_p50": q_price(t6m, 0.50),
        "module5_6m_p75": q_price(t6m, 0.75),
        "module5_6m_p95": q_price(t6m, 0.95),

        "generated_at_bjt": now_bjt_iso(),
    }
    return pred


# =========================
# 评估（昨日预测回顾）
# =========================
def load_last_pred_from_jsonl(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    # 取最后一行
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except Exception:
                continue
    return last

def calc_hit_band(actual: float, p25: float, p75: float) -> str:
    if actual < p25:
        return "LOW(<P25)"
    if actual > p75:
        return "HIGH(>P75)"
    return "MID(P25~P75)"

def build_eval_row(
    market_row: Dict[str, Any],
    last_pred: Optional[Dict[str, Any]],
    run_id: str
) -> Dict[str, Any]:
    """
    若没有昨日预测缓存，则 eval 用 null，并标注 reason，不做伪造回填
    """
    target_trade_date = str(market_row["trade_date"])
    actual_close = float(market_row["close"])

    if not last_pred:
        return {
            "eval_date": target_trade_date,
            "target_trade_date": target_trade_date,
            "symbol": SYMBOL,
            "actual_close": actual_close,
            "pred_ref_date": None,
            "pred_median_t1": None,
            "pred_p25_t1": None,
            "pred_p75_t1": None,
            "abs_error": None,
            "rel_error_pct": None,
            "hit_band": None,
            "amount_yi_hkd": float(market_row["amount_yi_hkd"]),
            "volume_percentile_20d": None,
            "enable_volume_factor": False,
            "enable_beta_anchor": False,
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "reason": "未找到昨日预测缓存（predictions/pred_daily.jsonl 为空或不存在），合规：不伪造昨日预测。"
        }

    # 兼容字段名
    pred_date = last_pred.get("pred_date")
    pred_median = safe_float(last_pred.get("module3_t1_p50"), None)
    pred_p25 = safe_float(last_pred.get("module3_t1_p25"), None)
    pred_p75 = safe_float(last_pred.get("module3_t1_p75"), None)

    if pred_median is None or pred_p25 is None or pred_p75 is None:
        return {
            "eval_date": target_trade_date,
            "target_trade_date": target_trade_date,
            "symbol": SYMBOL,
            "actual_close": actual_close,
            "pred_ref_date": pred_date,
            "pred_median_t1": pred_median,
            "pred_p25_t1": pred_p25,
            "pred_p75_t1": pred_p75,
            "abs_error": None,
            "rel_error_pct": None,
            "hit_band": None,
            "amount_yi_hkd": float(market_row["amount_yi_hkd"]),
            "volume_percentile_20d": None,
            "enable_volume_factor": bool(last_pred.get("enable_volume_factor", False)),
            "enable_beta_anchor": bool(last_pred.get("enable_beta_anchor", False)),
            "run_id": run_id,
            "model_version": MODEL_VERSION,
            "reason": "昨日预测记录缺少关键分位字段，合规：不计算误差。"
        }

    abs_error = abs(actual_close - pred_median)
    rel_error = pct(actual_close, pred_median)
    hit_band = calc_hit_band(actual_close, pred_p25, pred_p75)

    return {
        "eval_date": target_trade_date,
        "target_trade_date": target_trade_date,
        "symbol": SYMBOL,
        "actual_close": actual_close,
        "pred_ref_date": pred_date,
        "pred_median_t1": float(pred_median),
        "pred_p25_t1": float(pred_p25),
        "pred_p75_t1": float(pred_p75),
        "abs_error": float(abs_error),
        "rel_error_pct": float(rel_error),
        "hit_band": hit_band,
        "amount_yi_hkd": float(market_row["amount_yi_hkd"]),
        "volume_percentile_20d": None,
        "enable_volume_factor": bool(last_pred.get("enable_volume_factor", False)),
        "enable_beta_anchor": bool(last_pred.get("enable_beta_anchor", False)),
        "run_id": run_id,
        "model_version": MODEL_VERSION
    }


# =========================
# 报告（简单可读版，先保证输出）
# =========================
def fmt2(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def build_report(
    market_row: Dict[str, Any],
    eval_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    state: Dict[str, Any],
    run_log: Dict[str, Any],
    param_updates: List[Dict[str, Any]]
) -> str:
    d = market_row["trade_date"]
    lines: List[str] = []
    lines.append(f"# 京东物流 V1.0 预测报告（{d}）")
    lines.append("")
    lines.append(f"- 版本：{MODEL_VERSION}")
    lines.append(f"- 运行时间（BJT）：{now_bjt_iso()}")
    lines.append(f"- RunID：{run_log.get('run_id')}")
    lines.append("")

    # ① 当日实际交易数据
    lines.append("## ① 当日实际交易数据")
    lines.append(f"- 交易日：{market_row['trade_date']}")
    lines.append(f"- 开盘价：{fmt2(market_row['open'])}")
    lines.append(f"- 最高价：{fmt2(market_row['high'])}")
    lines.append(f"- 最低价：{fmt2(market_row['low'])}")
    lines.append(f"- 收盘价：{fmt2(market_row['close'])}")
    lines.append(f"- 成交量：{market_row['volume']}")
    lines.append(f"- 成交额：{fmt2(market_row['amount_yi_hkd'])}（亿港元）")
    lines.append(f"- 数据源：GitHub Raw JSON（锁定）")
    lines.append("")

    # ② 昨日预测回顾
    lines.append("## ② 昨日预测回顾")
    if eval_row.get("pred_ref_date") is None:
        lines.append(f"- 状态：无昨日预测缓存（合规：不伪造）")
        lines.append(f"- 说明：{eval_row.get('reason')}")
    else:
        lines.append(f"- 昨日预测日期：{eval_row.get('pred_ref_date')}")
        lines.append(f"- 昨日预测中位价（T+1）：{fmt2(eval_row.get('pred_median_t1'))}")
        lines.append(f"- 今日实际收盘价：{fmt2(eval_row.get('actual_close'))}")
        lines.append(f"- 绝对误差：{fmt2(eval_row.get('abs_error'))}")
        lines.append(f"- 相对误差：{fmt2(eval_row.get('rel_error_pct'))}%")
        lines.append(f"- 命中区间：{eval_row.get('hit_band')}")
    lines.append("")

    # ③ 次日价格分布预测
    lines.append("## ③ 次日价格分布预测（T+1）")
    lines.append(f"- P05：{fmt2(pred_row['module3_t1_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module3_t1_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module3_t1_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module3_t1_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module3_t1_p95'])}")
    lines.append("")

    # ④ 未来1个月
    lines.append("## ④ 未来1个月价格分布预测")
    lines.append(f"- P05：{fmt2(pred_row['module4_1m_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module4_1m_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module4_1m_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module4_1m_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module4_1m_p95'])}")
    lines.append("")

    # ⑤ 未来6个月
    lines.append("## ⑤ 未来6个月价格分布预测")
    lines.append(f"- P05：{fmt2(pred_row['module5_6m_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module5_6m_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module5_6m_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module5_6m_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module5_6m_p95'])}")
    lines.append("")

    # ⑥ 模型状态与学习更新
    lines.append("## ⑥ 模型状态与学习更新")
    lines.append(f"- sigma_base：{fmt2(state.get('sigma_base'))}")
    lines.append(f"- mu_base：{fmt2(state.get('mu_base'))}")
    if param_updates:
        lines.append(f"- 本次更新条目：{len(param_updates)}")
        for u in param_updates[-5:]:
            lines.append(f"  - {u.get('type')}：{u.get('old_sigma_base')} → {u.get('new_sigma_base')}（{u.get('note')}）")
    else:
        lines.append("- 本次无学习更新（或无昨日预测可回顾）。")
    lines.append("")

    # ⑦ 过去五个交易日回顾（保底：从 eval.jsonl 取最近5条）
    lines.append("## ⑦ 过去五个交易日预测命中回顾（最近5条）")
    rows = load_last_n_jsonl(EVAL_PATH, 5)
    if not rows:
        lines.append("- 暂无历史回顾数据。")
    else:
        lines.append("| 交易日 | 昨日预测日期 | 预测中位(T+1) | 实际收盘 | 误差% | 命中 |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(
                f"| {r.get('target_trade_date')} | {r.get('pred_ref_date') or '—'} | "
                f"{fmt2(r.get('pred_median_t1'))} | {fmt2(r.get('actual_close'))} | "
                f"{fmt2(r.get('rel_error_pct'))} | {r.get('hit_band') or '—'} |"
            )

    lines.append("")
    return "\n".join(lines)

def load_last_n_jsonl(path: str, n: int) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    buf: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except Exception:
                continue
    return buf[-n:]


# =========================
# 主流程
# =========================
def main() -> None:
    run_id = uuid.uuid4().hex[:16]
    run_log = {
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "symbol": SYMBOL,
        "started_at_bjt": now_bjt_iso(),
        "market_source": "github_raw_json",
        "market_url": MARKET_URL,
        "status": "START",
        "report_output_status": "PENDING",
        "report_reason": None,
    }

    # 1) 拉取当日真实数据（方案A阻塞重试）
    market_row, diag = fetch_market_row_blocking(MARKET_URL)
    run_log["market_fetch_diag"] = diag

    if market_row is None:
        run_log["status"] = "FAIL"
        run_log["finished_at_bjt"] = now_bjt_iso()
        run_log["report_output_status"] = "NO_OUTPUT"
        run_log["report_reason"] = f"当日真实数据不可用（已重试{diag.get('attempts')}次）：{diag.get('last_error')}"
        append_jsonl(RUN_LOG_PATH, run_log)
        print(run_log["report_reason"])
        # 合规：不推断/不回填
        return

    # 2) 读 state
    state = load_state()

    # 3) 读昨日预测（用于回顾）
    last_pred = load_last_pred_from_jsonl(PRED_PATH)

    # 4) 生成 eval（昨日预测回顾）
    eval_row = build_eval_row(market_row, last_pred, run_id)
    append_jsonl(EVAL_PATH, eval_row)

    # 5) 自学习更新（有条件才更新）
    state, param_updates = maybe_self_learn_and_update_state(
        state=state,
        today_trade_date=str(market_row["trade_date"]),
        today_close=float(market_row["close"]),
        eval_row=eval_row,
        run_id=run_id
    )
    for u in param_updates:
        append_jsonl(UPD_PATH, u)

    # 6) 生成预测并写入 pred_daily
    pred_row = make_predictions(state, float(market_row["close"]), str(market_row["trade_date"]))
    pred_row["run_id"] = run_id
    append_jsonl(PRED_PATH, pred_row)

    # 7) 更新并保存 state
    state["last_close"] = float(market_row["close"])
    state["last_trade_date"] = str(market_row["trade_date"])
    state["updated_at_bjt"] = now_bjt_iso()
    save_json(STATE_PATH, state)

    # 8) 生成报告并写入
    run_log["status"] = "OK"
    run_log["finished_at_bjt"] = now_bjt_iso()
    run_log["report_output_status"] = "OUTPUT"
    run_log["report_reason"] = "当日真实数据拉取成功；预测/写入完成。"
    append_jsonl(RUN_LOG_PATH, run_log)

    report = build_report(market_row, eval_row, pred_row, state, run_log, param_updates)
    ensure_dir_for_file(REPORT_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # Actions 日志输出
    print(report)


if __name__ == "__main__":
    main()
