# -*- coding: utf-8 -*-
"""
JD Logistics V1.0 Executor (standalone, repo-friendly)

功能（关键产物）：
- 拉取外部市场数据（DATA_SOURCE_URL，通常是 JD-data 仓库的 jd-logistics-latest.json）
- 写入 market/market_history.jsonl（追加）
- 计算/更新波动率状态 state/state.json
- 生成预测分位（T+1 / 1M / 6M）写入 predictions/predictions.jsonl（追加）
- 基于“上一条预测”做“昨日预测回顾”写入 evaluation/eval_history.jsonl（追加）
- 生成 report_latest.md（Markdown 报告）
- 生成 latest_report.json（给 HTML 前端用）
- 生成 runs/run_log.jsonl（运行日志）

约束：
- 禁止“估算/回填”外部真实数据：模块①严格使用外部 JSON 返回字段
- 若外部数据不可用：本次运行失败并给出明确原因（由 workflow 做重试更合规）
"""

import json
import os
import math
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# 0) 常量与路径
# =========================

MODEL_VERSION = os.getenv("MODEL_VERSION", "V1.0")
SYMBOL = os.getenv("SYMBOL", "02618.HK")
RUN_TYPE = os.getenv("RUN_TYPE", "AUTO")
DATA_SOURCE_URL = os.getenv(
    "DATA_SOURCE_URL",
    "https://raw.githubusercontent.com/njedu2023-prog/JD-data/main/jd-logistics-latest.json",
)

# 仓库根目录：scripts/executor_v1.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[1]

MARKET_DIR = REPO_ROOT / "market"
PRED_DIR = REPO_ROOT / "predictions"
EVAL_DIR = REPO_ROOT / "evaluation"
STATE_DIR = REPO_ROOT / "state"
RUNS_DIR = REPO_ROOT / "runs"

MARKET_PATH = MARKET_DIR / "market_history.jsonl"
PRED_PATH = PRED_DIR / "predictions.jsonl"
EVAL_PATH = EVAL_DIR / "eval_history.jsonl"
STATE_PATH = STATE_DIR / "state.json"
RUN_LOG_PATH = RUNS_DIR / "run_log.jsonl"

REPORT_PATH = REPO_ROOT / "report_latest.md"

# ✅ 修复：同时写入 root 与 runs，避免破坏旧引用，同时确保 runs/latest_report.json 更新
LATEST_REPORT_JSON_PATH_ROOT = REPO_ROOT / "latest_report.json"
LATEST_REPORT_JSON_PATH_RUNS = RUNS_DIR / "latest_report.json"


# =========================
# 1) 基础工具
# =========================

def now_bjt() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))

def now_bjt_iso() -> str:
    return now_bjt().strftime("%Y-%m-%dT%H:%M:%S%z")

def ensure_dir_for_file(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return float(s)
    except Exception:
        return None

def load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: Path, obj: Any) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    ensure_dir_for_file(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_last_n_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows[-n:]
    except Exception:
        return []

def load_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    rows = load_last_n_jsonl(path, 1)
    return rows[0] if rows else None


# =========================
# 2) 数据抓取（外部市场数据）
# =========================

def fetch_json(url: str, timeout: int = 25) -> Dict[str, Any]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "jdlog-v1-executor"})
    r.raise_for_status()
    return r.json()

def read_market_from_source(url: str) -> Dict[str, Any]:
    """
    只做“读取+结构化”，不做任何猜测/回填。
    期望外部 JSON 至少包含：trade_date, open, close, volume, turnover（或 amount）
    """
    raw = fetch_json(url)
    # 兼容一些常见字段命名
    trade_date = str(raw.get("trade_date") or raw.get("date") or raw.get("tradeDate") or "").strip()

    open_px = safe_float(raw.get("open") or raw.get("open_price") or raw.get("o"))
    close_px = safe_float(raw.get("close") or raw.get("close_price") or raw.get("c"))
    high_px = safe_float(raw.get("high") or raw.get("high_price") or raw.get("h"))
    low_px = safe_float(raw.get("low") or raw.get("low_price") or raw.get("l"))

    volume = safe_float(raw.get("volume") or raw.get("vol") or raw.get("v"))
    turnover = safe_float(raw.get("turnover") or raw.get("amount") or raw.get("amt") or raw.get("turnover_value"))

    # 严格检查：模块①必须有“最核心字段”
    missing = []
    if not trade_date:
        missing.append("trade_date")
    if open_px is None:
        missing.append("open")
    if close_px is None:
        missing.append("close")
    if volume is None:
        missing.append("volume")
    if turnover is None:
        missing.append("turnover/amount")

    if missing:
        raise RuntimeError(f"市场数据字段缺失：{missing}（外部源返回不完整，禁止本地回填/估算）")

    market_row = {
        "symbol": str(raw.get("symbol") or raw.get("code") or SYMBOL),
        "trade_date": trade_date,
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": volume,
        "turnover": turnover,
        "source_url": url,
        "fetched_at_bjt": now_bjt_iso(),
        "raw": raw,  # 保留原始回包（便于排错）；不用于“推断补全”
    }
    return market_row


# =========================
# 3) 波动率/漂移估计（轻量、可重复）
# =========================

def compute_log_returns(closes: List[float]) -> List[float]:
    rets = []
    for i in range(1, len(closes)):
        if closes[i-1] <= 0 or closes[i] <= 0:
            continue
        rets.append(math.log(closes[i] / closes[i-1]))
    return rets

def ewma_volatility_annualized(returns: List[float], lam: float = 0.94) -> Optional[float]:
    """
    EWMA 方差：s_t = lam*s_{t-1} + (1-lam)*r_{t-1}^2
    年化：sqrt(252) * sqrt(s_t)
    """
    if len(returns) < 10:
        return None
    s = returns[0] ** 2
    for r in returns[1:]:
        s = lam * s + (1.0 - lam) * (r ** 2)
    return math.sqrt(252.0 * s)

def mean_drift_daily(returns: List[float]) -> float:
    # 漂移尽量保守：用均值，但做截断，避免异常样本导致漂移失真
    if len(returns) < 10:
        return 0.0
    m = sum(returns) / len(returns)
    # 截断到 ±0.5%/日
    m = max(min(m, 0.005), -0.005)
    return m

def sigma_for_horizon(ann_sigma: float, days: int) -> float:
    # sigma_day = ann_sigma / sqrt(252)
    sig_d = ann_sigma / math.sqrt(252.0)
    return sig_d * math.sqrt(max(days, 1))


# =========================
# 4) 分位计算：对数正态近似（给分布输出用）
# =========================

def norm_inv(p: float) -> float:
    """
    近似标准正态分位函数（Acklam approximation）。
    够用：p in (0,1)
    """
    # 边界
    p = min(max(p, 1e-12), 1.0 - 1e-12)

    # Coefficients in rational approximations
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
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

def lognormal_quantile(s0: float, mu: float, sigma: float, z: float) -> float:
    # S = S0 * exp( (mu - 0.5*sigma^2) + sigma*z )
    return float(s0 * math.exp((mu - 0.5 * sigma * sigma) + sigma * z))

def build_distribution(s0: float, mu: float, sigma: float) -> Dict[str, float]:
    ps = [0.05, 0.25, 0.50, 0.75, 0.95]
    out = {}
    for p in ps:
        z = norm_inv(p)
        out[f"p{int(p*100):02d}"] = round(lognormal_quantile(s0, mu, sigma, z), 3)
    return out


# =========================
# 5) 命中分位判断（昨日预测回顾）
# =========================

def hit_band(actual_close: float, p05: float, p25: float, p50: float, p75: float, p95: float) -> str:
    if actual_close <= p05:
        return "≤P05"
    if actual_close <= p25:
        return "P05~P25"
    if actual_close <= p50:
        return "P25~P50"
    if actual_close <= p75:
        return "P50~P75"
    if actual_close <= p95:
        return "P75~P95"
    return ">P95"

def judgment_from_hitband(band: str) -> str:
    # “合规解释”：中间分位区间更合理，极端区间提示偏离
    if band in ("P25~P50", "P50~P75"):
        return "命中（中枢区间）"
    if band in ("P05~P25", "P75~P95"):
        return "可接受（边缘区间）"
    return "偏离（极端区间）"


# =========================
# 6) 报告生成（Markdown）
# =========================

def md_kv_table(rows: List[Tuple[str, Any]]) -> str:
    s = "| 字段 | 数值 |\n|---|---|\n"
    for k, v in rows:
        s += f"| {k} | {v} |\n"
    return s

def md_dist_table(title: str, dist: Dict[str, float]) -> str:
    return (
        f"**{title}**\n\n"
        "| 分位 | 价格(HKD) |\n|---|---|\n"
        f"| P05 | {dist['p05']} |\n"
        f"| P25 | {dist['p25']} |\n"
        f"| P50 | {dist['p50']} |\n"
        f"| P75 | {dist['p75']} |\n"
        f"| P95 | {dist['p95']} |\n"
    )

def build_report(
    market_row: Dict[str, Any],
    eval_row: Dict[str, Any],
    pred_row: Dict[str, Any],
    state: Dict[str, Any],
    run_log: Dict[str, Any],
) -> str:
    # ① 当日实际交易数据
    m = market_row
    module1 = md_kv_table([
        ("代码", m.get("symbol", SYMBOL)),
        ("交易日", m.get("trade_date", "")),
        ("开盘价", m.get("open", "")),
        ("最高价", m.get("high", "")),
        ("最低价", m.get("low", "")),
        ("收盘价", m.get("close", "")),
        ("成交量", m.get("volume", "")),
        ("成交额", m.get("turnover", "")),
        ("数据源", m.get("source_url", "")),
        ("抓取时间(BJT)", m.get("fetched_at_bjt", "")),
    ])

    # ② 昨日预测回顾
    e = eval_row or {}
    module2 = md_kv_table([
        ("昨日预测中位价(P50)", e.get("y_pred_p50", "")),
        ("今日实际收盘价", e.get("today_close", "")),
        ("误差(实际-预测)", e.get("error", "")),
        ("命中分位区间", e.get("hit_band", "")),
        ("判断结果", e.get("judgment", "")),
        ("回顾基于预测日期", e.get("y_pred_trade_date", "")),
    ])

    # ③ ④ ⑤ 分布
    t1 = {
        "p05": pred_row.get("module3_t1_p05"),
        "p25": pred_row.get("module3_t1_p25"),
        "p50": pred_row.get("module3_t1_p50"),
        "p75": pred_row.get("module3_t1_p75"),
        "p95": pred_row.get("module3_t1_p95"),
    }
    m1 = {
        "p05": pred_row.get("module4_1m_p05"),
        "p25": pred_row.get("module4_1m_p25"),
        "p50": pred_row.get("module4_1m_p50"),
        "p75": pred_row.get("module4_1m_p75"),
        "p95": pred_row.get("module4_1m_p95"),
    }
    m6 = {
        "p05": pred_row.get("module5_6m_p05"),
        "p25": pred_row.get("module5_6m_p25"),
        "p50": pred_row.get("module5_6m_p50"),
        "p75": pred_row.get("module5_6m_p75"),
        "p95": pred_row.get("module5_6m_p95"),
    }

    module3 = md_dist_table("③ 次日价格分布预测（T+1）", t1)
    module4 = md_dist_table("④ 未来1个月价格分布预测", m1)
    module5 = md_dist_table("⑤ 未来6个月价格分布预测", m6)

    # ⑥ 模型状态与学习更新
    st_rows = [
        ("系统版本", state.get("model_version", MODEL_VERSION)),
        ("运行类型", RUN_TYPE),
        ("本次运行时间(BJT)", run_log.get("generated_at_bjt", "")),
        ("EWMA年化波动率(估计)", state.get("sigma_ewma_1m_ann", "")),
        ("日漂移μ(估计)", state.get("mu_daily", "")),
        ("最近样本条数(用于估计)", state.get("return_samples", "")),
        ("备注", run_log.get("report_reason", "")),
    ]
    module6 = md_kv_table(st_rows)

    # ⑦ 过去五个交易日预测命中回顾（直接读 EVAL 的最后5条）
    last5 = load_last_n_jsonl(EVAL_PATH, 5)
    module7 = "| 预测日期 | 实际日期 | 预测P50 | 实际收盘 | 误差 | 命中区间 | 判断 |\n|---|---|---:|---:|---:|---|---|\n"
    for r in last5:
        module7 += (
            f"| {r.get('y_pred_trade_date','')} | {r.get('today_trade_date','')} | "
            f"{r.get('y_pred_p50','')} | {r.get('today_close','')} | {r.get('error','')} | "
            f"{r.get('hit_band','')} | {r.get('judgment','')} |\n"
        )

    header = (
        f"# 京东物流 {state.get('model_version', MODEL_VERSION)} 法定预测报告\n\n"
        f"- 代码：{m.get('symbol', SYMBOL)}\n"
        f"- 交易日：{m.get('trade_date','')}\n"
        f"- 报告生成时间(BJT)：{run_log.get('generated_at_bjt','')}\n"
        f"- 数据源：{m.get('source_url','')}\n\n"
    )

    body = (
        "## ① 当日实际交易数据\n\n" + module1 + "\n"
        "## ② 昨日预测回顾\n\n" + module2 + "\n"
        "## ③ 次日价格分布预测\n\n" + module3 + "\n"
        "## ④ 未来1个月价格分布预测\n\n" + module4 + "\n"
        "## ⑤ 未来6个月价格分布预测\n\n" + module5 + "\n"
        "## ⑥ 模型状态与学习更新\n\n" + module6 + "\n"
        "## ⑦ 过去五个交易日预测命中回顾\n\n" + module7 + "\n"
    )
    return header + body


# =========================
# 7) 主流程
# =========================

def main() -> None:
    run_log: Dict[str, Any] = {
        "generated_at_bjt": now_bjt_iso(),
        "symbol": SYMBOL,
        "run_type": RUN_TYPE,
        "model_version": MODEL_VERSION,
        "data_source_url": DATA_SOURCE_URL,
        "status": "START",
        "report_reason": "",
    }

    try:
        # 1) 拉取市场数据（严格：缺字段就失败）
        market_row = read_market_from_source(DATA_SOURCE_URL)

        # 2) 写入 market_history（去重：同 trade_date 不重复写）
        last_market = load_last_jsonl(MARKET_PATH)
        if last_market and str(last_market.get("trade_date", "")) == str(market_row.get("trade_date", "")):
            run_log["report_reason"] = "市场数据已是最新交易日（不重复追加market_history）"
        else:
            append_jsonl(MARKET_PATH, market_row)
            run_log["report_reason"] = "成功拉取并写入当日市场数据"

        # 3) 载入历史 close 用于估计
        market_tail = load_last_n_jsonl(MARKET_PATH, 260)  # 最多取近一年
        closes = []
        for r in market_tail:
            c = safe_float(r.get("close"))
            if c is not None:
                closes.append(c)

        returns = compute_log_returns(closes)
        ann_sigma = ewma_volatility_annualized(returns) or 0.35  # 若历史太少，给保守默认值（这是模型参数，不是“市场数据回填”）
        mu_d = mean_drift_daily(returns)
        run_log["sigma_source"] = "EWMA(returns)" if len(returns) >= 10 else "DEFAULT(0.35)"

        # 4) 更新 state
        state = load_json(STATE_PATH, {})
        state.update({
            "model_version": MODEL_VERSION,
            "symbol": SYMBOL,
            "updated_at_bjt": now_bjt_iso(),
            "sigma_ewma_1m_ann": round(float(ann_sigma), 4),
            "mu_daily": round(float(mu_d), 6),
            "return_samples": len(returns),
        })
        save_json(STATE_PATH, state)

        # 5) 生成预测分布（基于当日收盘价）
        s0 = float(market_row["close"])
        # T+1：1个交易日；1M：21；6M：126
        sig_t1 = sigma_for_horizon(ann_sigma, 1)
        sig_1m = sigma_for_horizon(ann_sigma, 21)
        sig_6m = sigma_for_horizon(ann_sigma, 126)

        dist_t1 = build_distribution(s0, mu_d * 1, sig_t1)
        dist_1m = build_distribution(s0, mu_d * 21, sig_1m)
        dist_6m = build_distribution(s0, mu_d * 126, sig_6m)

        pred_row = {
            "symbol": market_row.get("symbol", SYMBOL),
            "trade_date": market_row.get("trade_date", ""),
            "generated_at_bjt": now_bjt_iso(),
            "module3_t1_p05": dist_t1["p05"],
            "module3_t1_p25": dist_t1["p25"],
            "module3_t1_p50": dist_t1["p50"],
            "module3_t1_p75": dist_t1["p75"],
            "module3_t1_p95": dist_t1["p95"],
            "module4_1m_p05": dist_1m["p05"],
            "module4_1m_p25": dist_1m["p25"],
            "module4_1m_p50": dist_1m["p50"],
            "module4_1m_p75": dist_1m["p75"],
            "module4_1m_p95": dist_1m["p95"],
            "module5_6m_p05": dist_6m["p05"],
            "module5_6m_p25": dist_6m["p25"],
            "module5_6m_p50": dist_6m["p50"],
            "module5_6m_p75": dist_6m["p75"],
            "module5_6m_p95": dist_6m["p95"],
        }
        append_jsonl(PRED_PATH, pred_row)

        # 6) 昨日预测回顾：拿“上一条预测”对比今天 close
        prev_pred = None
        preds_tail = load_last_n_jsonl(PRED_PATH, 2)
        if len(preds_tail) == 2:
            # 最新一条是本次 pred_row，上一条是昨日/上次
            prev_pred = preds_tail[0]

        eval_row: Dict[str, Any] = {}
        if prev_pred and prev_pred.get("module3_t1_p50") is not None:
            today_close = float(market_row["close"])
            y_p50 = float(prev_pred["module3_t1_p50"])
            y_p05 = float(prev_pred["module3_t1_p05"])
            y_p25 = float(prev_pred["module3_t1_p25"])
            y_p75 = float(prev_pred["module3_t1_p75"])
            y_p95 = float(prev_pred["module3_t1_p95"])

            band = hit_band(today_close, y_p05, y_p25, y_p50, y_p75, y_p95)
            eval_row = {
                "symbol": market_row.get("symbol", SYMBOL),
                "y_pred_trade_date": prev_pred.get("trade_date", ""),
                "today_trade_date": market_row.get("trade_date", ""),
                "y_pred_p50": round(y_p50, 3),
                "today_close": round(today_close, 3),
                "error": round(today_close - y_p50, 3),
                "hit_band": band,
                "judgment": judgment_from_hitband(band),
                "generated_at_bjt": now_bjt_iso(),
            }
            append_jsonl(EVAL_PATH, eval_row)
        else:
            # 首次运行或历史不足：不给“伪回顾”
            eval_row = {
                "symbol": market_row.get("symbol", SYMBOL),
                "y_pred_trade_date": "",
                "today_trade_date": market_row.get("trade_date", ""),
                "y_pred_p50": "",
                "today_close": round(float(market_row["close"]), 3),
                "error": "",
                "hit_band": "",
                "judgment": "历史预测不足：本次不生成昨日回顾（避免伪造历史）",
                "generated_at_bjt": now_bjt_iso(),
            }

        # 7) 写 markdown 报告
        report = build_report(market_row, eval_row, pred_row, state, run_log)
        ensure_dir_for_file(REPORT_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)

        # 8) 写 latest_report.json（给 HTML 用）
        payload = {
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
            "module_6_model_state": {
                **state,
                "model_version": MODEL_VERSION,
                "symbol": SYMBOL,
            },
            "module_7_last5_hit": load_last_n_jsonl(EVAL_PATH, 5),
        }

        # ✅ 关键修复：写两份，确保 runs/latest_report.json 一定更新
        save_json(LATEST_REPORT_JSON_PATH_ROOT, payload)
        save_json(LATEST_REPORT_JSON_PATH_RUNS, payload)

        run_log["status"] = "SUCCESS"
        append_jsonl(RUN_LOG_PATH, run_log)

        print(report)

    except Exception as e:
        run_log["status"] = "FAILED"
        run_log["error"] = str(e)
        run_log["traceback"] = traceback.format_exc()
        append_jsonl(RUN_LOG_PATH, run_log)

        # 失败也输出一个“最小可读”信息，方便 Actions 日志直接看
        msg = (
            f"[FAILED] JD Logistics V1.0 executor\n"
            f"- time(BJT): {now_bjt_iso()}\n"
            f"- symbol: {SYMBOL}\n"
            f"- data_source_url: {DATA_SOURCE_URL}\n"
            f"- error: {str(e)}\n"
        )
        print(msg)
        raise


if __name__ == "__main__":
    main()
