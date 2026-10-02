"""P7 분석 선택에 대한 민감도 -- 계획서 `계획/P7_표본_강건성_비교.md` 를 발표용으로 단순화한 형태로 구현한다.

변형마다 **한 축만** 바꿔 P5 와 같은 분석(위험집합 비교, 층별 표준화 RR, 기업 이력 군집 부트스트랩)을 다시 한다.
탐지·결과·층 계산은 detect_cnt / validate_cnt 를 그대로 재사용하고, 여기서는 변형을 만들고 결과를 같은 표로 모은다.
  A  같은 문턱(Φ, n) 유지 : cap 51·100, 구간 하한·상한 지수, 활동량 하한 대안
  B  지표 단위가 바뀌어 문턱을 다시 구함 : I2(순서 코드 합) -- 주 규칙처럼 해석하지 않음
  E  입금 약화 정의 : 하위 10%·2.5%, 추적 3개월 (문턱은 P36 판정 가능 전체에서 같은 방식으로 다시 구함)
  F  n·k : n 25%·45%, k=3, 그리고 최종 n(35%)으로 ② 비교 기준·③ 창 길이 선택을 처음부터 다시
  G  표본·기간 : P35+(M0), 2024·2025 기준월
  D  비교군·표준화 : 한 번도 탐지되지 않은 기업, 층 대안
"""
from __future__ import annotations

import math

import numpy as np

import cntlib as cl
import detect_cnt as dc
import validate_cnt as vc

MIN_DET = 30


# --------------------------------------------------------------------------- 활동량 지수 변형
def i1_bound(panel: cl.CountPanel, mask: np.ndarray, which: str, cap_high: float = 100.0) -> np.ndarray:
    """I1 을 구간의 하한('low') 또는 상한('high', 50건초과 = cap_high 시나리오)으로 합산한 지수. 미관측 달은 NaN."""
    obs = panel.observed[mask]
    return sum(cl.bound_count(cl.codes_float(panel.cnt[c][mask], obs), which, cap_high) for c in cl.CH5)


def i2_matrix(panel: cl.CountPanel, mask: np.ndarray) -> np.ndarray:
    """I2 = 국내 5채널 순서 코드(0~9)의 합. 구간 중앙값·50건초과 cap 가정에서 자유롭다. 미관측 달은 NaN."""
    obs = panel.observed[mask]
    return sum(cl.codes_float(panel.cnt[c][mask], obs) for c in cl.CH5)


# --------------------------------------------------------------------------- 탐지 일치도
def detection_overlap(det_a: np.ndarray, det_b: np.ndarray) -> dict:
    """두 규칙의 최초 탐지 기업 일치도: 기업 집합의 Jaccard, 같은 기업의 탐지월 차이(개월) 중앙값·절댓값 평균."""
    a, b = det_a >= 0, det_b >= 0
    inter, union = int((a & b).sum()), int((a | b).sum())
    both = a & b
    diff = (det_b[both] - det_a[both]).astype(float)
    return {"jaccard": inter / union if union else math.nan, "n_a": int(a.sum()), "n_b": int(b.sum()), "n_both": inter,
            "month_diff_median": float(np.median(diff)) if diff.size else math.nan,
            "same_month_share": float((diff == 0).mean()) if diff.size else math.nan}


# --------------------------------------------------------------------------- 최종 n 으로 ② 비교 기준·③ 창 길이를 처음부터 다시
def redo_selection(panel: cl.CountPanel, mask: np.ndarray, thr: float) -> dict:
    """임시 감소 기준을 thr 로 바꿔 ② 비교 기준(직전/전년)·③ 창 길이 k 선택 절차를 다시 돌리고(건수만 사용, 입금 미사용),
    ④ 활동량 하한 Φ(천장 칸 제외)·⑥ n 까지 새로 정한다."""
    I = dc.i1_matrix(panel, mask)
    sea = dc.seasonality(I)
    cm = dc.choose_mode(I, sea, thr=thr)
    mode = cm["mode"]
    shares = [dc.transient_share(I, k, mode, thr=thr) for k in dc.KS]
    k, why = dc.choose_k(shares)
    FL = dc.ceiling_window_flag(panel, mask, k, dc.lag_for(mode, k))
    d = dc.decide_fixed(I, mode, k, FL)
    return {"thr": thr, "mode": mode, "k": int(k), "why": why, "phi": float(d["phi"]), "n": float(d["n"]), "shares": shares}


# --------------------------------------------------------------------------- 변형 하나의 결과
def _first_detect_overlap(base_D: dict, D: dict):
    if len(base_D["det"]) != len(D["det"]):
        return None
    return detection_overlap(base_D["det"], D["det"])


def run_variant(D: dict, F: int, B: int = 2000, seed: int = 20261301, base: dict = None, cells=None, units=None) -> dict:
    """준비된 표본 D(vc.prepare 결과)에서 H-C1 RR 과 95% 범위. base(기준 결과)를 주면 '변형 RR ÷ 기준 RR'(같은 기업 재표집으로 짝지은 반복)도 구한다."""
    U = D["units"] if units is None else units
    R = vc.analyze_units(U, F, B, seed, cells=cells, keep_all=True)
    out = {"n_det": int(U.is_det.sum()), "R": R, "descriptive": int(U.is_det.sum()) < MIN_DET}
    if base is not None and not out["descriptive"]:
        sa, sb = base["R"]["star_all"], R["star_all"]          # 반복 번호가 같은 재표집끼리 짝짓는다 (정의되지 않은 반복은 뺀다)
        ok = np.isfinite(sa) & np.isfinite(sb) & (sa > 0)
        if sa.size == sb.size and ok.sum() >= 20:
            r = sb[ok] / sa[ok]
            out["ratio"] = (float(R["rr"] / base["R"]["rr"]), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))
    if base is not None and "D" in base and D is not base["D"]:
        out["overlap"] = _first_detect_overlap(base["D"], D)
    return out


def cells_pooled_months(u: vc.Units, n_sector: int = 6):
    """층 대안 1: 기준월을 풀링한 규모 5분위 x 업종군 (병합용 상위 층 = 규모)."""
    return vc.make_cells(u.size * n_sector + u.sector, u.size.astype(np.int64))


def cells_no_sector(u: vc.Units):
    """층 대안 2: 업종군을 쓰지 않는 기준월 x 규모 5분위."""
    raw = (u.e * 5 + u.size).astype(np.int64)
    return vc.make_cells(raw, raw)


def never_detected_controls(u: vc.Units, det_firm: np.ndarray) -> vc.Units:
    """보조 비교군: 탐지 기업과 '한 번도 탐지되지 않은 기업'의 기업-월만 (이후에 탐지될 기업은 비교군에서 뺀다 → 미래 정보로 선택된 안정 기업이라 RR 이 부풀 수 있음)."""
    keep = u.is_det | (det_firm[u.firm] < 0)
    return u.sub(keep)
