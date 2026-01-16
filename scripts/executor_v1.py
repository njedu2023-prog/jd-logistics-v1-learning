# -*- coding: utf-8 -*-

"""
JD Logistics V1.0 - Executor (EWMA + Student-t + Term Structure)

实现你确认的 3 件事（最小可落地版本）：
1) EWMA 预测波动率 sigma(t)   -> 输出 sigma_1m / sigma_6m（年化）
2) 收益分布用 t 分布（厚尾） -> 用 nu 自由度；把 t 分布缩放到单位方差
3) 1M / 6M 分开 sigma（期限结构） -> 1M 用 sigma_1m，6M 用 sigma_6m（T+1 用 sigma_1m）

说明：
- 为了能计算 EWMA，需要历史收益序列。本文件新增写入：market/market_history.jsonl
  每天把真实收盘价追加进去，然后用最近 N 天计算收益并滚动更新 sigma。
"""

import os
import json
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

# ==============================
# 从 GitHub Secrets 读取港股 API Token
# ==============================

HK_MARKET_API_TOKEN = os.getenv("HK_MARKET_API_TOKEN")
if not HK_MARKET_API_TOKEN:
    raise RuntimeError("❌ 未读取到 HK_MARKET_API_TOKEN，请检查 GitHub Secrets 配置")

# =========================
# 固定配置
# =========================

MODEL_VERSION = "V1.0"
SYMBOL = "02618.HK"

# 模块①唯一数据源（当前仍是 GitHub Raw JSON）
MARKET_URL = "https://raw.githubusercontent.com/njedu2023-prog/jd-logistics-v1-learning/main/jd-logistics-latest.json"

# 输出路径（仓库内）
RUN_LOG_PATH = "runs/run_log.jsonl"
EVAL_PATH = "evaluation/eval.jsonl"
UPD_PATH = "learning/updates.jsonl"
PRED_PATH = "predictions/pred_daily.jsonl"
STATE_PATH = "state/model_state.json"
REPORT_PATH = "report_latest.md"
LATEST_REPORT_JSON_PATH = "runs/latest_report.json"

# 新增：保存真实历史（用于 EWMA）
MARKET_HIST_PATH = "market/market_history.jsonl"

# 抓取策略：有限次阻塞重试
RETRY_MAX = 12
RETRY_SLEEP_SEC = 10
HTTP_TIMEOUT = 15

# 交易日换算
TRADING_DAYS_PER_YEAR = 252
DT_1D = 1.0 / TRADING_DAYS_PER_YEAR
DT_1M = 21.0 / TRADING_DAYS_PER_YEAR
DT_6M = 126.0 / TRADING_DAYS_PER_YEAR

# EWMA 参数（你后面要更“量化”可以继续调）
# 经验：短期波动更敏感 -> lambda 小一点；长期更平滑 -> lambda 大一点
EWMA_LAMBDA_1M = 0.94
EWMA_LAMBDA_6M = 0.97

# 估计波动用的最大历史长度（越大越稳，但越慢）
MAX_RET_DAYS = 260  # 约 1 年

# t 分布自由度（越小越厚尾；必须 > 2 才有方差）
DEFAULT_T_NU = 7.0


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

def pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return abs(a - b) / abs(b) * 100.0


# =========================
# 数据抓取（阻塞重试）
# =========================
def http_get_text(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests 未安装。请在 requirements.txt 里加入 requests。")
    r = requests.get(url, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.text

def fetch_market_row_blocking(url: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
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

            market = dict(raw)

            # date -> trade_date
            if "trade_date" not in market and "date" in market:
                market["trade_date"] = market.get("date")

            # amount(HKD) -> amount_yi_hkd(亿HKD)
            if "amount_yi_hkd" not in market and "amount" in market:
                amt = market.get("amount", None)
                if amt is None:
                    raise ValueError("amount为空")
                market["amount_yi_hkd"] = round(float(amt) / 1e8, 4)

            required = ["trade_date", "open", "high", "low", "close", "volume", "amount_yi_hkd"]
            missing = [k for k in required if k not in market or market[k] in ("", None)]
            if missing:
                raise ValueError(f"JSON缺字段或为空: {missing}")

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
# 历史市场数据（用于 EWMA）
# =========================
def append_market_history(market_row: Dict[str, Any]) -> None:
    """每天把真实市场数据追加到 market_history.jsonl，供波动率估计使用。"""
    rec = {
        "symbol": SYMBOL,
        "trade_date": market_row.get("trade_date"),
        "close": float(market_row.get("close")),
        "open": float(market_row.get("open")),
        "high": float(market_row.get("high")),
        "low": float(market_row.get("low")),
        "volume": int(market_row.get("volume")),
        "amount_yi_hkd": float(market_row.get("amount_yi_hkd")),
        "fetched_at_bjt": market_row.get("fetched_at_bjt"),
        "raw_sha1": market_row.get("raw_sha1"),
    }
    append_jsonl(MARKET_HIST_PATH, rec)

def load_last_n_market_history(n: int = MAX_RET_DAYS + 2) -> List[Dict[str, Any]]:
    if not os.path.exists(MARKET_HIST_PATH):
        return []
    buf: List[Dict[str, Any]] = []
    with open(MARKET_HIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                buf.append(json.loads(line))
            except Exception:
                continue
    return buf[-n:]


def compute_log_returns_from_history(hist: List[Dict[str, Any]]) -> List[float]:
    """用 close 计算 log return 序列（按时间顺序）"""
    if not hist or len(hist) < 2:
        return []
    closes: List[float] = []
    for x in hist:
        c = safe_float(x.get("close"), None)
        if c is None or c <= 0:
            continue
        closes.append(float(c))
    if len(closes) < 2:
        return []
    rets: List[float] = []
    for i in range(1, len(closes)):
        rets.append(math.log(closes[i] / closes[i - 1]))
    return rets


# =========================
# 状态（新增：sigma_1m / sigma_6m / nu）
# =========================
def default_state() -> Dict[str, Any]:
    return {
        "model_version": MODEL_VERSION,
        "created_at_bjt": now_bjt_iso(),
        "updated_at_bjt": now_bjt_iso(),

        # 旧字段保留（兼容前端/报告）
        "sigma_base": 0.035,
        "mu_base": 0.0,

        # 新字段：期限结构（年化）
        "sigma_1m": 0.35,
        "sigma_6m": 0.40,

        # 厚尾 t 分布参数
        "t_nu": DEFAULT_T_NU,

        "last_close": None,
        "last_trade_date": None,

        # 显示兼容（前端已有字段映射）
        "sigma_short": 0.35,
        "sigma_mid": 0.35,
        "sigma_long": 0.40,
        "enable_volume_factor": False,
        "enable_beta_anchor": False,
    }

def load_state() -> Dict[str, Any]:
    st = read_json(STATE_PATH, default=None)
    if not isinstance(st, dict):
        st = default_state()
        save_json(STATE_PATH, st)

    # 补齐缺失字段
    for k, v in default_state().items():
        if k not in st:
            st[k] = v
    return st


# =========================
# EWMA 波动率估计（年化）
# =========================
def ewma_annualized_sigma(returns: List[float], lam: float, init_sigma: float) -> float:
    """
    returns: 日频 log return
    lam: EWMA lambda
    init_sigma: 初始年化sigma（用于历史不足时）
    输出：年化 sigma
    """
    if not returns:
        return float(init_sigma)

    # 把年化 sigma 转成日方差作为初值
    var = (init_sigma * init_sigma) / TRADING_DAYS_PER_YEAR

    # 用 EWMA 更新日方差
    for r in returns:
        var = lam * var + (1.0 - lam) * (r * r)

    # 转回年化
    ann = math.sqrt(max(var, 1e-12) * TRADING_DAYS_PER_YEAR)
    return float(min(max(ann, 0.01), 2.0))  # 夹一下，避免极端


def update_sigmas_from_history(state: Dict[str, Any], market_hist: List[Dict[str, Any]], run_id: str) -> List[Dict[str, Any]]:
    """
    用历史收益滚动更新 sigma_1m / sigma_6m，并写入 updates.jsonl（可追溯）
    """
    updates: List[Dict[str, Any]] = []

    returns = compute_log_returns_from_history(market_hist)
    if not returns:
        return updates

    # 只用最近 MAX_RET_DAYS 天
    returns = returns[-MAX_RET_DAYS:]

    old_1m = float(state.get("sigma_1m", state.get("sigma_short", 0.35)))
    old_6m = float(state.get("sigma_6m", state.get("sigma_long", 0.40)))

    new_1m = ewma_annualized_sigma(returns, EWMA_LAMBDA_1M, old_1m)
    new_6m = ewma_annualized_sigma(returns, EWMA_LAMBDA_6M, old_6m)

    # 写回 state（并同步兼容字段）
    state["sigma_1m"] = float(new_1m)
    state["sigma_6m"] = float(new_6m)

    state["sigma_short"] = float(new_1m)
    state["sigma_mid"] = float(new_1m)
    state["sigma_long"] = float(new_6m)

    # 兼容旧字段（系统里若有人还用 sigma_base）
    state["sigma_base"] = float(new_1m)

    updates.append({
        "update_time_bjt": now_bjt_iso(),
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "type": "ewma_sigma_term_structure",
        "lambda_1m": EWMA_LAMBDA_1M,
        "lambda_6m": EWMA_LAMBDA_6M,
        "ret_days_used": len(returns),
        "old_sigma_1m": old_1m,
        "new_sigma_1m": float(new_1m),
        "old_sigma_6m": old_6m,
        "new_sigma_6m": float(new_6m),
        "note": "EWMA 更新：生成 1M/6M 年化波动率期限结构（t 分布预测将使用它）"
    })
    return updates


# =========================
# 分布：t 分布分位（缩放到单位方差）
# =========================
def _normal_quantile(p: float) -> float:
    # 你原来的近似实现（保留）
    p = min(max(p, 1e-10), 1 - 1e-10)
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

def t_quantile_unitvar(p: float, nu: float) -> float:
    """
    返回：t 分布分位点，经过缩放，使其方差=1（便于 sigma 直接当波动率用）
    - 若 scipy 可用：用 scipy.stats.t.ppf
    - 否则：退化为正态近似（nu 大时误差小；nu=7 时会偏保守）
    """
    nu = float(nu)
    if nu <= 2.0:
        nu = 3.0

    # 单位方差缩放因子：Var(t_nu)=nu/(nu-2)
    scale = math.sqrt(nu / (nu - 2.0))

    # 尝试 scipy
    try:
        from scipy.stats import t as student_t  # type: ignore
        q = float(student_t.ppf(p, df=nu))
        return q / scale
    except Exception:
        # 无 scipy：用正态近似兜底
        return _normal_quantile(p)


# =========================
# 预测（t 分布 + 1M/6M sigma）
# =========================
def q_price_log_t(spot: float, mu: float, sigma_ann: float, t_year: float, p: float, nu: float) -> float:
    """
    对数收益：
      r = (mu - 0.5*sigma^2)*t + sigma*sqrt(t)*Z
    其中 Z ~ t(nu) 并缩放到单位方差
    """
    z = t_quantile_unitvar(p, nu)
    return float(spot * math.exp((mu - 0.5 * sigma_ann * sigma_ann) * t_year + sigma_ann * math.sqrt(t_year) * z))

def make_predictions(state: Dict[str, Any], spot_close: float, target_trade_date: str) -> Dict[str, Any]:
    mu = float(0.0 if state.get("mu_base") is None else state.get("mu_base"))
    nu = float(state.get("t_nu", DEFAULT_T_NU))

    sigma_1m = float(state.get("sigma_1m", state.get("sigma_short", 0.35)))
    sigma_6m = float(state.get("sigma_6m", state.get("sigma_long", 0.40)))

    # 规则：T+1 用 1M sigma；1M 用 1M sigma；6M 用 6M sigma
    def q(t: float, p: float, sig: float) -> float:
        return q_price_log_t(spot_close, mu, sig, t, p, nu)

    return {
        "pred_date": target_trade_date,
        "symbol": SYMBOL,
        "model_version": MODEL_VERSION,

        # 把核心参数写进预测记录（便于审计）
        "mu_base": mu,
        "t_nu": nu,
        "sigma_1m": sigma_1m,
        "sigma_6m": sigma_6m,
        "sigma_short": sigma_1m,
        "sigma_long": sigma_6m,
        "enable_volume_factor": False,
        "enable_beta_anchor": False,

        # ③ 次日（T+1）
        "module3_t1_p05": q(DT_1D, 0.05, sigma_1m),
        "module3_t1_p25": q(DT_1D, 0.25, sigma_1m),
        "module3_t1_p50": q(DT_1D, 0.50, sigma_1m),
        "module3_t1_p75": q(DT_1D, 0.75, sigma_1m),
        "module3_t1_p95": q(DT_1D, 0.95, sigma_1m),

        # ④ 1M
        "module4_1m_p05": q(DT_1M, 0.05, sigma_1m),
        "module4_1m_p25": q(DT_1M, 0.25, sigma_1m),
        "module4_1m_p50": q(DT_1M, 0.50, sigma_1m),
        "module4_1m_p75": q(DT_1M, 0.75, sigma_1m),
        "module4_1m_p95": q(DT_1M, 0.95, sigma_1m),

        # ⑤ 6M
        "module5_6m_p05": q(DT_6M, 0.05, sigma_6m),
        "module5_6m_p25": q(DT_6M, 0.25, sigma_6m),
        "module5_6m_p50": q(DT_6M, 0.50, sigma_6m),
        "module5_6m_p75": q(DT_6M, 0.75, sigma_6m),
        "module5_6m_p95": q(DT_6M, 0.95, sigma_6m),

        "generated_at_bjt": now_bjt_iso(),
    }


# =========================
# 评估（昨日预测回顾）
# =========================
def load_last_pred_from_jsonl(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
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

def build_eval_row(market_row: Dict[str, Any], last_pred: Optional[Dict[str, Any]], run_id: str) -> Dict[str, Any]:
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
# 报告（先保证输出）
# =========================
def fmt2(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

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

    lines.append("## ① 当日实际交易数据")
    lines.append(f"- 交易日：{market_row['trade_date']}")
    lines.append(f"- 开盘价：{fmt2(market_row['open'])}")
    lines.append(f"- 最高价：{fmt2(market_row['high'])}")
    lines.append(f"- 最低价：{fmt2(market_row['low'])}")
    lines.append(f"- 收盘价：{fmt2(market_row['close'])}")
    lines.append(f"- 成交量：{market_row['volume']}")
    lines.append(f"- 成交额：{fmt2(market_row['amount_yi_hkd'])}（亿港元）")
    lines.append("- 数据源：GitHub Raw JSON（锁定）")
    lines.append("")

    lines.append("## ② 昨日预测回顾")
    if eval_row.get("pred_ref_date") is None:
        lines.append("- 状态：无昨日预测缓存（合规：不伪造）")
        lines.append(f"- 说明：{eval_row.get('reason')}")
    else:
        lines.append(f"- 昨日预测日期：{eval_row.get('pred_ref_date')}")
        lines.append(f"- 昨日预测中位价（T+1）：{fmt2(eval_row.get('pred_median_t1'))}")
        lines.append(f"- 今日实际收盘价：{fmt2(eval_row.get('actual_close'))}")
        lines.append(f"- 绝对误差：{fmt2(eval_row.get('abs_error'))}")
        lines.append(f"- 相对误差：{fmt2(eval_row.get('rel_error_pct'))}%")
        lines.append(f"- 命中区间：{eval_row.get('hit_band')}")
    lines.append("")

    lines.append("## ③ 次日价格分布预测（T+1）")
    lines.append(f"- P05：{fmt2(pred_row['module3_t1_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module3_t1_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module3_t1_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module3_t1_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module3_t1_p95'])}")
    lines.append("")

    lines.append("## ④ 未来1个月价格分布预测（sigma_1m）")
    lines.append(f"- P05：{fmt2(pred_row['module4_1m_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module4_1m_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module4_1m_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module4_1m_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module4_1m_p95'])}")
    lines.append("")

    lines.append("## ⑤ 未来6个月价格分布预测（sigma_6m）")
    lines.append(f"- P05：{fmt2(pred_row['module5_6m_p05'])}")
    lines.append(f"- P25：{fmt2(pred_row['module5_6m_p25'])}")
    lines.append(f"- P50：{fmt2(pred_row['module5_6m_p50'])}")
    lines.append(f"- P75：{fmt2(pred_row['module5_6m_p75'])}")
    lines.append(f"- P95：{fmt2(pred_row['module5_6m_p95'])}")
    lines.append("")

    lines.append("## ⑥ 模型状态与学习更新")
    lines.append(f"- sigma_1m（年化）：{fmt2(state.get('sigma_1m'))}")
    lines.append(f"- sigma_6m（年化）：{fmt2(state.get('sigma_6m'))}")
    lines.append(f"- t 分布自由度 nu：{fmt2(state.get('t_nu'))}")
    if param_updates:
        lines.append(f"- 本次更新条目：{len(param_updates)}")
        for u in param_updates[-5:]:
            lines.append(f"  - {u.get('type')}：1M {u.get('old_sigma_1m')}→{u.get('new_sigma_1m')}；6M {u.get('old_sigma_6m')}→{u.get('new_sigma_6m')}")
    else:
        lines.append("- 本次无更新（历史不足或首次运行）。")
    lines.append("")

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

def build_placeholder_report(run_log: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 京东物流 V1.0 预测报告（占位）")
    lines.append("")
    lines.append(f"- 版本：{MODEL_VERSION}")
    lines.append(f"- 运行时间（BJT）：{now_bjt_iso()}")
    lines.append(f"- RunID：{run_log.get('run_id')}")
    lines.append(f"- 状态：FAIL（模块①真实数据不可用）")
    lines.append("")
    lines.append("## 失败原因")
    lines.append(f"- {run_log.get('report_reason')}")
    lines.append("")
    lines.append("## 输出说明（合规）")
    lines.append("- 本次未获取到当日真实交易数据，因此：")
    lines.append("  - 不生成预测分布（避免伪造）")
    lines.append("  - 不生成昨日回顾（避免伪造）")
    lines.append("  - 仅输出占位报告与法定结构 latest_report.json，保证界面可用且可追溯")
    lines.append("")
    return "\n".join(lines)

def write_placeholder_latest_report_json(run_log: Dict[str, Any]) -> None:
    obj = {
        "status": "FAIL",
        "symbol": SYMBOL,
        "trade_date": "",
        "generated_at_bjt": now_bjt_iso(),
        "report_reason": str(run_log.get("report_reason") or ""),
        "module_1_market": {},
        "module_2_yesterday_review": {},
        "module_3_t1_distribution": [],
        "module_4_1m_distribution": [],
        "module_5_6m_distribution": [],
        "module_6_model_state": {},
        "module_7_last5_hit": []
    }
    save_json(LATEST_REPORT_JSON_PATH, obj)


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

    market_row, diag = fetch_market_row_blocking(MARKET_URL)
    run_log["market_fetch_diag"] = diag

    if market_row is None:
        run_log["status"] = "FAIL"
        run_log["finished_at_bjt"] = now_bjt_iso()
        run_log["report_output_status"] = "OUTPUT_PLACEHOLDER"
        run_log["report_reason"] = f"当日真实数据不可用（已重试{diag.get('attempts')}次）：{diag.get('last_error')}"
        append_jsonl(RUN_LOG_PATH, run_log)

        placeholder_md = build_placeholder_report(run_log)
        ensure_dir_for_file(REPORT_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(placeholder_md)

        write_placeholder_latest_report_json(run_log)

        print(run_log["report_reason"])
        print(placeholder_md)
        return

    # 1) 先把今日真实市场数据写入历史（供 EWMA 用）
    append_market_history(market_row)

    # 2) 读状态 & 用历史收益更新 sigma_1m / sigma_6m
    state = load_state()
    market_hist = load_last_n_market_history()
    sigma_updates = update_sigmas_from_history(state, market_hist, run_id=run_id)
    for u in sigma_updates:
        append_jsonl(UPD_PATH, u)

    # 3) 昨日预测回顾
    last_pred = load_last_pred_from_jsonl(PRED_PATH)
    eval_row = build_eval_row(market_row, last_pred, run_id)
    append_jsonl(EVAL_PATH, eval_row)

    # 4) 生成今天预测（t 分布 + 期限结构 sigma）
    pred_row = make_predictions(state, float(market_row["close"]), str(market_row["trade_date"]))
    pred_row["run_id"] = run_id
    append_jsonl(PRED_PATH, pred_row)

    # 5) 更新 state
    state["last_close"] = float(market_row["close"])
    state["last_trade_date"] = str(market_row["trade_date"])
    state["updated_at_bjt"] = now_bjt_iso()
    save_json(STATE_PATH, state)

    # 6) 写 run_log
    run_log["status"] = "OK"
    run_log["finished_at_bjt"] = now_bjt_iso()
    run_log["report_output_status"] = "OUTPUT"
    run_log["report_reason"] = "当日真实数据拉取成功；EWMA更新sigma；t分布预测写入完成。"
    append_jsonl(RUN_LOG_PATH, run_log)

    # 7) 写 markdown 报告
    report = build_report(market_row, eval_row, pred_row, state, run_log, sigma_updates)
    ensure_dir_for_file(REPORT_PATH)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # 8) 写 latest_report.json（给 HTML 用）
    save_json(LATEST_REPORT_JSON_PATH, {
        "status": "SUCCESS",
        "symbol": SYMBOL,
        "trade_date": str(market_row.get("trade_date", "")),
        "generated_at_bjt": now_bjt_iso(),
        "report_reason": run_log["report_reason"],
        "module_1_market": market_row,
        "module_2_yesterday_review": eval_row,

        "module_3_t1_distribution": [
            {"p": 0.05, "v": pred_row.get("module3_t1_p05")},
            {"p": 0.25, "v": pred_row.get("module3_t1_p25")},
            {"p": 0.50, "v": pred_row.get("module3_t1_p50")},
            {"p": 0.75, "v": pred_row.get("module3_t1_p75")},
            {"p": 0.95, "v": pred_row.get("module3_t1_p95")},
        ],
        "module_4_1m_distribution": [
            {"p": 0.05, "v": pred_row.get("module4_1m_p05")},
            {"p": 0.25, "v": pred_row.get("module4_1m_p25")},
            {"p": 0.50, "v": pred_row.get("module4_1m_p50")},
            {"p": 0.75, "v": pred_row.get("module4_1m_p75")},
            {"p": 0.95, "v": pred_row.get("module4_1m_p95")},
        ],
        "module_5_6m_distribution": [
            {"p": 0.05, "v": pred_row.get("module5_6m_p05")},
            {"p": 0.25, "v": pred_row.get("module5_6m_p25")},
            {"p": 0.50, "v": pred_row.get("module5_6m_p50")},
            {"p": 0.75, "v": pred_row.get("module5_6m_p75")},
            {"p": 0.95, "v": pred_row.get("module5_6m_p95")},
        ],

        # module_6_model_state：把模型关键参数都放进去（HTML 前端会自动翻中文）
        "module_6_model_state": {
            **state,
            "model_version": MODEL_VERSION,
            "symbol": SYMBOL,
        },

        "module_7_last5_hit": load_last_n_jsonl(EVAL_PATH, 5),
    })

    print(report)


if __name__ == "__main__":
    main()
