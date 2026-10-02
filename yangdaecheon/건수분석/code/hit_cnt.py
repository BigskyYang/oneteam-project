"""P9 보강 검증(노트북 08) -- 계획서 `계획/P9_보강검증_적중과_기준기간.md`.

  1) 신호가 얼마나 맞히고 놓치는가: 신호/비신호 x 약화/비약화 2x2 표(기업-월 단위, P5 의 RR 과 같은 단위), 기준(n)별 적중·놓침, 기업 단위 신호 시점
  2) 기준 기간을 공유하지 않는 신호: 신호를 '전년 같은 달 대비 감소'로 정의하면 신호의 비교 달이 요구불입금액 약화의 기준 달과 겹치지 않는다

탐지·결과·층·표준화는 detect_cnt / validate_cnt 를 그대로 재사용하고, 여기서는 표를 만드는 집계만 한다. 화면 출력은 집계값뿐이다.
"""
from __future__ import annotations

import math

import numpy as np

import detect_cnt as dc
import validate_cnt as vc

EPS = 1e-9


def _div(a: float, b: float) -> float:
    return a / b if b else math.nan


# --------------------------------------------------------------------------- 1) 기업-월 단위 2x2
def two_by_two(u) -> dict:
    """신호(최초 탐지 기업-월) / 비신호(위험집합 기업-월) x 약화 / 비약화 칸 수와 비율.

    a = 신호 & 약화, b = 신호 & 비약화, c = 비신호 & 약화, d = 비신호 & 비약화.
    precision(적중 비율) = a/(a+b), ctrl_rate = c/(c+d), recall(약화 기업-월 중 신호가 잡은 비율) = a/(a+c), miss_share = c/(a+c).
    단위가 기업-월이라 같은 기업이 여러 기준월에 나올 수 있다(P5 의 RR 과 같은 단위)."""
    det, ev = np.asarray(u.is_det, bool), np.asarray(u.ev, bool)
    a, b = int((det & ev).sum()), int((det & ~ev).sum())
    c, d = int((~det & ev).sum()), int((~det & ~ev).sum())
    prec, ctrl = _div(a, a + b), _div(c, c + d)
    return {"a": a, "b": b, "c": c, "d": d, "n_signal": a + b, "n_other": c + d, "precision": prec, "ctrl_rate": ctrl,
            "recall": _div(a, a + c), "miss_share": _div(c, a + c), "lift_raw": _div(prec, ctrl) if ctrl and ctrl == ctrl else math.nan}


def monthly_averages(t: dict, n_months: int) -> dict:
    """기준월 평균: 월평균 신호 기업, 월평균 약화 기업-월(신호+비신호), 그중 신호가 잡은 수."""
    return {"signals_per_month": t["n_signal"] / n_months, "weak_per_month": (t["a"] + t["c"]) / n_months, "caught_per_month": t["a"] / n_months}


# --------------------------------------------------------------------------- 1b) 기업 단위: 신호가 약화보다 먼저였나
def weak_matrix(D: dict) -> np.ndarray:
    """(n, 36) bool: 기준월 e 에 판정 가능하고 결과 g 가 정의되며 g <= θ 이면 약화."""
    st, G, th = D["status"], D["G"], D["theta"]
    W = np.zeros(st.shape, dtype=bool)
    for e in D["er"]:
        W[:, e] = (st[:, e] == 0) & ~np.isnan(G[:, e]) & (G[:, e] <= th + EPS)
    return W


def firm_timing(D: dict) -> dict:
    """기업 단위 요약(시점 구분 포함).

    약화 기업 = 판정 가능 기준월 중 한 번이라도 약화(g <= θ)가 있었던 기업, 첫 약화 기준월 e*.
    신호가 먼저 = 신호 기준월 <= e*, 신호가 늦음 = 신호 기준월 > e*, 신호 없음 = 신호가 한 번도 없던 기업."""
    er = list(D["er"])
    lo, hi = er[0], er[-1]
    W = weak_matrix(D)[:, lo : hi + 1]
    judge_any = (D["status"][:, lo : hi + 1] == 0).any(axis=1)
    weak = W.any(axis=1)
    first = np.where(weak, W.argmax(axis=1) + lo, -1)
    det = D["det"]
    sig = det >= 0
    early = weak & sig & (det <= first)
    late = weak & sig & (det > first)
    none = weak & ~sig
    return {"n_judgeable": int(judge_any.sum()), "n_weak": int(weak.sum()), "n_signal": int(sig.sum()), "n_signal_weak": int((sig & weak).sum()),
            "early": int(early.sum()), "late": int(late.sum()), "none": int(none.sum())}


# --------------------------------------------------------------------------- 1c) 기준(n)별 적중·놓침
def precision_recall_grid(panel, mask, consts: dict, n_list) -> list[dict]:
    """감소 기준 n 만 바꿔(Φ·비교 기준·k·θ·경계는 P36 에서 고정) 2x2 표와 점추정 RR 을 낸다."""
    rows = []
    for n in n_list:
        D = vc.prepare(panel, mask, consts, n_pct=float(n))
        u = D["units"]
        t = two_by_two(u)
        cu, cg = vc.cells_main(u)
        est = vc.estimate(u, cu, cg)
        rows.append({"n": float(n), **t, "rr": est["rr"], "p_ctrl_std": est["p_ctrl"], "n_months": len(D["er"]), "n_detected": int((D["det"] >= 0).sum())})
    return rows


# --------------------------------------------------------------------------- 2) 기준 기간을 공유하지 않는 신호
def reference_overlap(k: int, L: int, e: int = 20) -> dict:
    """기준월 e 에서 신호의 비교 달(창 [e-k+1, e] 의 L 개월 전)과 요구불입금액 약화의 기준 달 [e-11, e-6] 이 겹치는 달 수."""
    ref = set(range(e - k + 1 - L, e + 1 - L))
    base = set(range(e - 11, e - 5))
    ov = sorted(ref & base)
    return {"signal_reference": [min(ref), max(ref)], "outcome_base": [min(base), max(base)], "overlap_months": len(ov)}


def year_on_year_rule(panel, mask, K: int = 6):
    """신호를 '전년 같은 달 대비 감소'(L=12)로 정의한 규칙. 활동량 하한 Φ 와 감소 기준 n 을 이 규칙에서 다시 구한다(P4 v2 와 같은 절차:
    천장 칸을 뺀 분위별 IQR 로 Φ, 변화율 5번째 백분위수로 n). 반환: (결정 dict, 고정 상수 dict)."""
    I = dc.i1_matrix(panel, mask)
    FL = dc.ceiling_window_flag(panel, mask, K, dc.lag_for("전년", K))
    d = dc.decide_fixed(I, "전년", K, FL)
    consts = vc.fit_constants(panel, mask, "전년", K, d["phi"], d["n"])
    return d, consts
