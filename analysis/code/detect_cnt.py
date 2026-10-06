"""P4 활동 위축 탐지 규칙 -- 계획서 `계획/P4_활동위축_탐지규칙.md` 3·4·4b·4d 의 수식을 그대로 구현한다.

입금은 이 모듈에서 쓰지 않는다 (P5 의 검증이 순환이 되지 않도록). 월 인덱스는 0 기준(0 = 2023.01 ... 35 = 2025.12)이고
계획서의 후보월 t(1~36) 는 e = t - 1 이다. 후보월 범위는 t ∈ [k+12, 30] 즉 e ∈ [k+11, 29].
"""
from __future__ import annotations

import math
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd

import cntlib as cl

T = 36
E_MAX = 29            # 마지막 후보월 e (= 30번째 달): 사후 6개월 추적이 필요하다
KS = (1, 2, 3, 6)     # 창 길이 후보
MODES = ("전년", "직전")  # 비교 기준: 작년 같은 달 / 직전 기간
PROV_THR = 0.30       # 임시 감소 기준: 비교 기준·창 길이를 정할 때만 쓴다
TRANS_THR = -0.10     # 일시적 하락: 사후 k개월 변화율이 이 값 이상이면 회복
TRANS_SHARE = 0.20    # 일시적 하락 비율이 이 값 미만인 가장 짧은 창
EPS = 1e-9


# --------------------------------------------------------------------------- 지수와 변화율
def i1_matrix(panel: cl.CountPanel, mask: np.ndarray, cap: float = cl.DEFAULT_CAP) -> np.ndarray:
    """총 활동량 I1 = 국내 5채널 근사 건수 합 (n, 36). 미관측 달은 NaN."""
    Z = [cl.codes_float(panel.cnt[c][mask], panel.observed[mask]) for c in cl.CH5]
    return sum(cl.approx_count(z, cap) for z in Z)


def lag_for(mode: str, k: int) -> int:
    """비교 간격 L: 전년 비교 12, 직전 기간 비교 k."""
    if mode not in MODES:
        raise ValueError(mode)
    return 12 if mode == "전년" else int(k)


def e_lo_for(k: int, L: int, e_lo: int | None = None) -> int:
    """첫 후보월. 기본은 k+11 (계획서 ④-4: 전년 비교(L=12)를 가정해 정한 범위를 두 비교 방식에 똑같이 적용). 직전 기간 비교(L=k)만 보면 변화율이
    정의되는 가장 이른 달 k-1+L (k=6 이면 11 = 2023.12)까지 내릴 수 있다. P10 민감도(후보월 확장)에서만 e_lo 를 준다."""
    lo = k + 11 if e_lo is None else int(e_lo)
    if lo < k - 1 + L:
        raise cl.CntError(f"첫 후보월 {lo} 는 변화율이 정의되는 가장 이른 달 {k - 1 + L} 보다 이를 수 없습니다.")
    return lo


def e_range(k: int, e_lo: int | None = None) -> range:
    return range(k + 11 if e_lo is None else int(e_lo), E_MAX + 1)


def window_change(I: np.ndarray, e: int, k: int, L: int, need: int):
    """창 [e-k+1, e] 와 L 개월 전 창에서 **두 시점 모두 관측된 달만** 짝지어 평균한다 (M0 쌍 맞춤).

    반환 (R, C, pairs): R = 최근 평균, C = 비교 평균, pairs = 짝지은 달 수. 짝이 하나도 없으면 NaN.
    need 미만 여부는 호출 쪽에서 판정한다.
    """
    n = I.shape[0]
    a = e - k + 1
    if a - L < 0 or e >= I.shape[1]:
        nan = np.full(n, np.nan)
        return nan, nan.copy(), np.zeros(n, dtype=int)
    cur, old = I[:, a : e + 1], I[:, a - L : e + 1 - L]
    ok = ~np.isnan(cur) & ~np.isnan(old)
    pairs = ok.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        R = np.nanmean(np.where(ok, cur, np.nan), axis=1)
        C = np.nanmean(np.where(ok, old, np.nan), axis=1)
    return R, C, pairs


class Change(NamedTuple):
    r: np.ndarray      # (n, 36) 변화율, 정의되지 않으면 NaN
    C: np.ndarray      # (n, 36) 비교 평균 (짝지은 달이 need 미만이면 NaN, C=0 이면 0)
    cnt: np.ndarray    # (n, 36) 짝지은 달 수
    need: int
    k: int
    L: int


def change_matrix(I: np.ndarray, k: int, L: int, need: int | None = None) -> Change:
    """모든 e 에 대해 r(i,e) = (R-C)/C. C>0 이고 짝지은 달 >= need 일 때만 정의."""
    need = cl.min_pairs(k) if need is None else int(need)
    n, T_ = I.shape
    r = np.full((n, T_), np.nan)
    C = np.full((n, T_), np.nan)
    cnt = np.zeros((n, T_), dtype=np.int16)
    for e in range(k - 1 + L, T_):
        R_, C_, c_ = window_change(I, e, k, L, need)
        okp = c_ >= need
        C[:, e] = np.where(okp, C_, np.nan)
        cnt[:, e] = c_
        with np.errstate(divide="ignore", invalid="ignore"):
            r[:, e] = np.where(okp & (C_ > 0), (R_ - C_) / C_, np.nan)
    return Change(r=r, C=C, cnt=cnt, need=need, k=int(k), L=int(L))


def post_change(I: np.ndarray, k: int) -> np.ndarray:
    """사후 변화율 r_post(i,e) = 창 [e+1, e+k] 를 **작년 같은 달**(L=12) 과 비교한 변화율 -> (n, 36) 에서 열 e 에 놓는다.

    비교 방식과 무관하게 L=12 를 쓴다: 직전 기간 비교로 L=k 를 쓰면 '탐지 창 대비 추가 하락 여부'가 되어 회복을 재지 못한다.
    """
    ch = change_matrix(I, k, 12)
    out = np.full_like(ch.r, np.nan)
    out[:, : T - k] = ch.r[:, k:]
    return out


# --------------------------------------------------------------------------- 판정 상태 / 탐지
def status_matrix(ch: Change, phi: float, e_lo: int | None = None) -> np.ndarray:
    """(n, 36) int8: -1 판정 기간 밖, 0 판정 가능, 1 (a) 비교 평균 0, 2 (b) 활동량 하한 미만, 3 (c) 빈 달 부족."""
    n, T_ = ch.r.shape
    st = np.full((n, T_), -1, dtype=np.int8)
    lo = e_lo_for(ch.k, ch.L, e_lo)
    sl = slice(lo, E_MAX + 1)
    c_bad = ch.cnt[:, sl] < ch.need
    a_bad = ~c_bad & (ch.C[:, sl] <= 0)
    with np.errstate(invalid="ignore"):
        b_bad = ~c_bad & ~a_bad & (ch.C[:, sl] < phi)
    s = np.zeros(c_bad.shape, dtype=np.int8)
    s[a_bad] = 1
    s[b_bad] = 2
    s[c_bad] = 3
    st[:, sl] = s
    return st


def first_detect(ch: Change, phi: float, thr: float, e_lo: int | None = None) -> np.ndarray:
    """판정 기간에서 r <= -thr 가 처음 성립하는 달 e (기업당 1건). 없으면 -1. thr 는 양의 비율(0.3 = 30%)."""
    lo = e_lo_for(ch.k, ch.L, e_lo)
    r, C = ch.r[:, lo : E_MAX + 1], ch.C[:, lo : E_MAX + 1]
    with np.errstate(invalid="ignore"):
        cond = ~np.isnan(r) & (C >= phi) & (r <= -thr + EPS)
    has = cond.any(axis=1)
    return np.where(has, cond.argmax(axis=1) + lo, -1)


def first_detect_m1(panel: cl.CountPanel, mask: np.ndarray, k: int, L: int, phi: float, thr: float,
                    cap: float = cl.DEFAULT_CAP) -> np.ndarray:
    """M1(인과적 채움): 후보월 e 마다 월 <= e 의 관측값만으로 과거 빈 달을 채워 순차적으로 재계산한 최초 탐지월.

    각 후보월의 채움값과 판정은 그 시점에 고정된다(이후 관측으로 소급 갱신하지 않음). 문턱(phi, thr)은 호출 쪽에서 준 사후 고정 값.
    """
    Z = [cl.codes_float(panel.cnt[c][mask], panel.observed[mask]) for c in cl.CH5]
    n = Z[0].shape[0]
    need = cl.min_pairs(k)
    det = np.full(n, -1, dtype=int)
    for e in e_range(k):
        Is = sum(cl.approx_count(cl.fill_causal(z, e), cap) for z in Z)
        R, C, pairs = window_change(Is, e, k, L, need)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (R - C) / C
            ok = (pairs >= need) & (C > 0) & (C >= phi) & (r <= -thr + EPS)
        det = np.where(ok & (det < 0), e, det)
    return det


# --------------------------------------------------------------------------- ① 계절성, ② 비교 기준
def seasonality(I: np.ndarray) -> dict:
    """월 지수 = 그 달 기업 I1 중앙값 ÷ 그 해 월 중앙값 평균. 연도 쌍 상관 평균 >= 0.5 and 연도별 최대/최소 평균 >= 1.3 이면 '있음'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(I, axis=0)
    years = [med[12 * y : 12 * y + 12] for y in range(3)]
    idx = [y / np.nanmean(y) for y in years]
    cors = [float(np.corrcoef(idx[a], idx[b])[0, 1]) for a, b in ((0, 1), (1, 2), (0, 2))]
    mm = [float(np.nanmax(v) / np.nanmin(v)) for v in idx]
    corr_mean, mm_mean = float(np.mean(cors)), float(np.mean(mm))
    return {"corr_mean": corr_mean, "maxmin_mean": mm_mean, "flag": bool(corr_mean >= 0.5 and mm_mean >= 1.3),
            "index": [v.round(3).tolist() for v in idx]}


def monthly_rates(I: np.ndarray, k: int, L: int, thr: float = PROV_THR) -> np.ndarray:
    """후보월별 탐지율 = 변화율이 정의된 기업 중 r <= -thr 인 비율 (Φ 미적용: 임시 단계)."""
    ch = change_matrix(I, k, L)
    lo = k + 11
    r = ch.r[:, lo : E_MAX + 1]
    ok = ~np.isnan(r)
    with np.errstate(invalid="ignore"):
        hit = ok & (r <= -thr + EPS)
    return hit.sum(axis=0) / np.maximum(ok.sum(axis=0), 1)


def choose_mode(I: np.ndarray, sea: dict, k: int = 3, thr: float | None = None) -> dict:
    thr = PROV_THR if thr is None else thr          # P7: 최종 n 으로 선택 절차를 다시 돌릴 때만 바꾼다
    rates = {m: monthly_rates(I, k, lag_for(m, k), thr) for m in MODES}
    ratio = {m: (float(v.max() / v.min()) if v.min() > 0 else math.inf) for m, v in rates.items()}
    if sea["flag"]:
        mode = min(MODES, key=lambda m: ratio[m])
    else:
        mode = "직전"
    return {"mode": mode, "ratio": ratio, "rates": {m: v.round(4).tolist() for m, v in rates.items()}}


# --------------------------------------------------------------------------- ③ 창 길이 k
def transient_share(I: np.ndarray, k: int, mode: str, thr: float | None = None) -> dict:
    """임시 기준(30%)으로 처음 탐지된 기업 중 사후 k개월 변화율(작년 같은 달 대비) >= -10% 인 비율."""
    L = lag_for(mode, k)
    ch = change_matrix(I, k, L)
    first = first_detect(ch, 0.0, PROV_THR if thr is None else thr)
    idx = np.flatnonzero(first >= 0)
    rp = post_change(I, k)[idx, first[idx]]
    okp = ~np.isnan(rp)
    d, tr = int(okp.sum()), int((rp[okp] >= TRANS_THR).sum())
    return {"k": k, "detected": int(len(idx)), "defined": d, "transient": tr, "share": (tr / d) if d else math.nan}


def _norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def lower_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """H1: 두 번째 비율이 첫 번째보다 낮다 (한쪽 검정 p)."""
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else _norm_cdf((p2 - p1) / se)


def choose_k(shares: list[dict]) -> tuple[int, str]:
    """일시적 하락 비율이 20% 미만인 가장 짧은 창. 모두 20% 이상이면 인접 창 비율 검정으로 더 이상 줄지 않는 창."""
    ok = [s for s in shares if s["defined"] > 0]
    if not ok:
        raise cl.CntError("모든 창에서 탐지 기업이 없어 창 길이를 정할 수 없습니다.")
    for s in ok:
        if s["share"] < TRANS_SHARE:
            return s["k"], f"일시적 하락 비율 {s['share'] * 100:.1f}% < 20%인 가장 짧은 창"
    for a, b in zip(ok[:-1], ok[1:]):
        if lower_p(a["transient"], a["defined"], b["transient"], b["defined"]) >= 0.05:
            return a["k"], "모두 20% 이상: 다음 창이 유의하게 낮지 않은 창"
    return ok[-1]["k"], "모두 20% 이상: 비율이 계속 줄어 가장 긴 창"


# --------------------------------------------------------------------------- ④ 활동량 하한 Φ
def _merge_bins(c_sorted_bins: np.ndarray, n_bins: int, min_bin: int) -> np.ndarray:
    """분위당 min_bin 미만이면 인접 분위(위쪽, 마지막이면 아래쪽)와 병합. 병합 후 분위 번호를 0..B-1 로 다시 매긴다."""
    ids = c_sorted_bins.copy()
    while True:
        u, cnt = np.unique(ids, return_counts=True)
        small = np.flatnonzero(cnt < min_bin)
        if len(u) <= 1 or small.size == 0:
            break
        j = int(small[0])
        tgt = u[j + 1] if j + 1 < len(u) else u[j - 1]
        ids[ids == u[j]] = tgt
    _, ids = np.unique(ids, return_inverse=True)
    return ids


def decile_floor(C: np.ndarray, r: np.ndarray, min_bin: int = 100, n_bins: int = 10) -> tuple[float, dict]:
    """활동량 하한 Φ (계획서 4b). C>0 이고 r 이 정의된 (i,e) 만 대상.

    C 를 값 기준 10분위로 나누되 경계가 같으면 합치고, 분위당 min_bin 미만이면 인접 분위와 병합한다. 분위별 IQR(r) 을 구하고
    ref = 상위 절반(10분위면 상위 5개) 분위 IQR 의 중앙값. 모든 d >= d* 에서 IQR_d <= 1.5 x ref 인 가장 낮은 d* 의 하한값이 Φ (d*=0 이면 0).
    """
    ok = ~np.isnan(r) & (C > 0)
    c, rr = C[ok], r[ok]
    if c.size < min_bin:
        raise cl.CntError("활동량 하한을 정할 관측이 부족합니다.")
    edges = np.unique(np.quantile(c, np.linspace(0, 1, n_bins + 1)[1:-1]))
    b0 = np.searchsorted(edges, c, side="right")
    ids = _merge_bins(b0, n_bins, min_bin)
    B = int(ids.max()) + 1
    iqr = np.array([np.subtract(*np.percentile(rr[ids == d], [75, 25])) for d in range(B)])
    cnt = np.array([(ids == d).sum() for d in range(B)])
    lo = np.array([c[ids == d].min() for d in range(B)])
    hi = np.array([c[ids == d].max() for d in range(B)])
    top = math.ceil(B / 2)
    ref = float(np.median(iqr[B - top :]))
    passes = iqr <= 1.5 * ref + EPS
    d_star = None
    for d in range(B):
        if passes[d:].all():
            d_star = d
            break
    if d_star is None:
        raise cl.CntError("활동량 하한 규칙을 만족하는 분위가 없습니다 (변화율 분포를 확인하세요).")
    phi = 0.0 if d_star == 0 else float(lo[d_star])
    return phi, {"bins": B, "d_star": d_star, "ref": ref, "iqr": iqr, "count": cnt, "lo": lo, "hi": hi, "passes": passes}


def phi_bootstrap(firm: np.ndarray, C: np.ndarray, r: np.ndarray, B: int = 200, seed: int = 11, min_bin: int = 100) -> np.ndarray:
    """기업 단위 재표집으로 Φ 가 얼마나 흔들리는지 본다 (보고용, Φ 자체는 바꾸지 않는다). firm: 각 (i,e) 쌍의 기업 번호 0..F-1."""
    ok = ~np.isnan(r) & (C > 0)
    firm, C, r = firm[ok], C[ok], r[ok]
    F = int(firm.max()) + 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        mult = np.bincount(rng.integers(0, F, F), minlength=F)
        w = mult[firm]
        sel = np.repeat(np.arange(firm.size), w)
        try:
            out.append(decile_floor(C[sel], r[sel], min_bin=min_bin)[0])
        except cl.CntError:
            out.append(math.nan)
    return np.array(out)


# --------------------------------------------------------------------------- ⑥ n
def n_from_r(r_judgeable: np.ndarray) -> dict:
    """n0 = -100 x Q(0.05) (양의 %), n = max(5, 5%p 단위 반올림(0.5 올림)). n0 <= 0 이면 임계를 정의할 수 없어 멈춘다."""
    r = r_judgeable[~np.isnan(r_judgeable)]
    if r.size == 0:
        raise cl.CntError("판정 가능한 변화율이 없습니다.")
    q05 = float(np.percentile(r, 5))
    n0 = -100.0 * q05
    if n0 <= 0:
        raise cl.CntError("변화율의 5번째 백분위수가 0 이상이라 탐지 임계를 정의할 수 없습니다. 분포를 확인하세요.")
    n = max(5.0, math.floor(n0 / 5.0 + 0.5) * 5.0)
    exact = float((np.abs(r + n / 100.0) < EPS).mean())
    return {"q05": q05, "n0": n0, "n": n, "exact_share": exact}


def describe_r(r: np.ndarray) -> dict:
    r = r[~np.isnan(r)]
    q = np.percentile(r, [1, 5, 10, 25, 50, 75, 90])
    sd = float(r.std(ddof=1))
    skew = float(((r - r.mean()) ** 3).mean() / (r.std() ** 3)) if r.std() > 0 else math.nan
    return {"관측": int(r.size), "평균": float(r.mean()), "표준편차": sd, "1%": q[0], "5%": q[1], "10%": q[2], "25%": q[3],
            "50%": q[4], "75%": q[5], "90%": q[6], "왜도": skew}


# --------------------------------------------------------------------------- 전체 결정 절차 (①~⑥)
def decide_fixed(I: np.ndarray, mode: str, k: int, floor_exclude: np.ndarray | None = None) -> dict:
    """④Φ → ⑤판정 가능 집합 → ⑥n 만: 비교 기준과 k 가 주어졌을 때의 결정.

    floor_exclude: (n, 36) bool. True 인 칸은 **Φ(분위별 IQR) 산출에서만** 뺀다 (판정 가능·n 산출은 모든 칸). 진단용 대안 하한에 쓴다.
    """
    L = lag_for(mode, k)
    ch = change_matrix(I, k, L)
    lo = k + 11
    Cw, rw = ch.C[:, lo : E_MAX + 1], ch.r[:, lo : E_MAX + 1]
    r_floor = rw if floor_exclude is None else np.where(floor_exclude[:, lo : E_MAX + 1], np.nan, rw)
    phi, info = decile_floor(Cw, r_floor)
    judge = ~np.isnan(rw) & (Cw >= phi)
    nn = n_from_r(np.where(judge, rw, np.nan))
    return {"mode": mode, "k": k, "L": L, "phi": phi, "phi_info": info, "n": nn["n"], "n_info": nn, "judge_pairs": int(judge.sum()), "change": ch}


def decide(I: np.ndarray) -> dict:
    """한 표본에서 ①계절성 → ②비교 기준 → ③창 길이 k → ④Φ → ⑤판정 가능 집합 → ⑥n 을 순서대로 정한다."""
    sea = seasonality(I)
    cm = choose_mode(I, sea)
    mode = cm["mode"]
    shares = [transient_share(I, k, mode) for k in KS]
    k, why = choose_k(shares)
    d = decide_fixed(I, mode, k)
    d.update({"sea": sea, "mode_info": cm, "shares": shares, "k_why": why})
    return d


def apply_rule(I: np.ndarray, mode: str, k: int, phi: float, n: float, e_lo: int | None = None) -> dict:
    """고정된 규칙(비교 기준, k, Φ, n)을 다른 표본에 그대로 적용해 최초 탐지월과 판정 상태를 얻는다. e_lo: 첫 후보월(기본 k+11, P10 에서만 바꿈)."""
    ch = change_matrix(I, k, lag_for(mode, k))
    det = first_detect(ch, phi, n / 100.0, e_lo)
    st = status_matrix(ch, phi, e_lo)
    return {"det": det, "status": st, "change": ch}


# --------------------------------------------------------------------------- 검정력 (계획서 4d)
def _se(N1: float, N0: float, rr: float, p0: float, deff: float) -> float:
    p1 = rr * p0
    return math.sqrt(deff * ((1 - p1) / (N1 * p1) + (1 - p0) / (N0 * p0)))


def mde(N1: int, N0: int, p0: float = 0.05, deff: float = 1.5, z: float = 2.8) -> float:
    """80% 검정력·양측 5% 에서 ln(RR) >= 2.8 x SE 가 되는 가장 작은 RR. 탐색 상한은 탐지군 약화율 p1 = RR x p0 <= 50% 이며,
    그 안에서도 못 찾으면 inf (p1 이 1 에 가까우면 분산이 0 이 되는 수학적 허점을 피한다).
    """
    if N1 <= 0 or N0 <= 0:
        return math.inf
    hi = 0.5 / p0

    def gap(rr):
        return math.log(rr) - z * _se(N1, N0, rr, p0, deff)

    if gap(hi) < 0:
        return math.inf
    lo = 1.0001
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return hi


def expected_ci(N1: int, N0: int, rr: float, p0: float = 0.05, deff: float = 1.5) -> tuple[float, float, float]:
    """(95% CI 하한, 상한, 탐지군 기대 약화 사건 수)."""
    se = _se(N1, N0, rr, p0, deff)
    return rr * math.exp(-1.96 * se), rr * math.exp(1.96 * se), N1 * rr * p0


# --------------------------------------------------------------------------- 보고용 표
def first_table(I: np.ndarray, label: str, p0: float = 0.05, deff: float = 1.5, ceil_flag_fn=None) -> pd.DataFrame:
    """k별·비교 방식별: 최대 후보월 수, 판정 가능 기업-월 수, 판정 불가 3종, 탐지 기업 수(임시 30%, 최종 n), N0, MDE.

    ceil_flag_fn(k, L) -> (n, 36) bool 을 주면 그 칸을 Φ(분위별 IQR) 산출에서 뺀다 (수정 규칙 v2).
    """
    rows = []
    for mode in MODES:
        for k in KS:
            ch = change_matrix(I, k, lag_for(mode, k))
            lo = k + 11
            Cw, rw = ch.C[:, lo : E_MAX + 1], ch.r[:, lo : E_MAX + 1]
            rf = rw if ceil_flag_fn is None else np.where(ceil_flag_fn(k, lag_for(mode, k))[:, lo : E_MAX + 1], np.nan, rw)
            try:
                phi, _ = decile_floor(Cw, rf)
            except cl.CntError:
                phi = math.nan
            st = status_matrix(ch, 0.0 if math.isnan(phi) else phi)[:, lo : E_MAX + 1]
            tot = st.size
            judge = (st == 0)
            n_row = {"표본": label, "비교 기준": mode, "k": k, "최대 후보월 수": 19 - k, "Φ": phi,
                     "판정 가능 기업-월": int(judge.sum()), "판정 가능 %": 100 * judge.sum() / tot,
                     "(a) 비교 평균 0 %": 100 * (st == 1).sum() / tot, "(b) 하한 미만 %": 100 * (st == 2).sum() / tot,
                     "(c) 빈 달 부족 %": 100 * (st == 3).sum() / tot}
            try:
                nn = n_from_r(np.where(judge, rw, np.nan))
                n_use = nn["n"]
            except cl.CntError:
                nn, n_use = None, math.nan
            d30 = first_detect(ch, 0.0 if math.isnan(phi) else phi, PROV_THR)
            n_row["N1 (임시 30%)"] = int((d30 >= 0).sum())
            n_row["n"] = n_use
            if nn is not None:
                dn = first_detect(ch, phi, n_use / 100.0)
                n1 = int((dn >= 0).sum())
                n_judge_firms = int(judge.any(axis=1).sum())
                n0 = max(n_judge_firms - n1, 0)
                n_row.update({"N1 (최종 n)": n1, "판정 가능 기업": n_judge_firms, "N0": n0, "MDE": mde(n1, n0, p0, deff)})
            rows.append(n_row)
    return pd.DataFrame(rows)


def ceiling_at_first_judgeable(panel: cl.CountPanel, mask: np.ndarray, ch: Change, phi: float) -> np.ndarray:
    """기업의 **첫 판정 가능월** 비교 창에 50건초과(코드 9) 채널이 하나라도 있었는가 (탐지 결과와 무관하게 정의). 판정 가능월이 없으면 False."""
    codes = np.stack([panel.cnt[c][mask] for c in cl.CH5])        # (5, n, 36)
    obs = panel.observed[mask]
    any9 = ((codes == 9) & obs[None]).any(axis=0)               # (n, 36)
    lo = ch.k + 11
    with np.errstate(invalid="ignore"):
        judge = ~np.isnan(ch.r[:, lo : E_MAX + 1]) & (ch.C[:, lo : E_MAX + 1] >= phi)
    has = judge.any(axis=1)
    first = np.where(has, judge.argmax(axis=1) + lo, -1)
    out = np.zeros(len(first), dtype=bool)
    for i in np.flatnonzero(has):
        e = first[i]
        a = e - ch.k + 1 - ch.L
        out[i] = any9[i, max(a, 0) : e + 1 - ch.L].any()
    return out


def firm_attribute(panel: cl.CountPanel, mask: np.ndarray, name: str) -> np.ndarray:
    """기업별 속성 코드(첫 관측값), 없으면 -1."""
    A = panel.attr[name][mask]
    out = np.full(A.shape[0], -1, dtype=int)
    for j in range(A.shape[1] - 1, -1, -1):          # 뒤에서부터 덮어써 첫 관측값이 남게 한다
        m = A[:, j] >= 0
        out[m] = A[m, j]
    return out


# --------------------------------------------------------------------------- 진단 (해석용, 사전 규칙이 아님)
def ceiling_window_flag(panel: cl.CountPanel, mask: np.ndarray, k: int, L: int) -> np.ndarray:
    """(n, 36) bool: 후보월 e 의 최근 창 [e-k+1, e] 또는 비교 창 [e-k+1-L, e-L] 에 50건초과(코드 9) 채널이 있었는가."""
    codes = np.stack([panel.cnt[c][mask] for c in cl.CH5])
    obs = panel.observed[mask]
    any9 = ((codes == 9) & obs[None]).any(axis=0)
    n, T_ = any9.shape
    out = np.zeros((n, T_), dtype=bool)
    for e in range(k - 1 + L, T_):
        a = e - k + 1
        out[:, e] = any9[:, a : e + 1].any(axis=1) | any9[:, a - L : e + 1 - L].any(axis=1)
    return out


def decompose_detection(panel: cl.CountPanel, mask: np.ndarray, det: np.ndarray, k: int, L: int, cap: float = cl.DEFAULT_CAP) -> np.ndarray:
    """탐지월에서 총 활동량 변화(R - C)를 채널별로 나눠, **비교 창에서 50건초과(코드 9)가 절반 이상이던 채널(천장 채널)** 이 설명하는 몫을 구한다.

    반환: 탐지된 기업 순서대로 길이 m 의 배열 [-1 = 천장 채널 없음, 그 밖에는 천장 채널의 변화 / 총 변화 (0~1 이면 천장 채널이 하락을 설명)].
    """
    Z = [cl.codes_float(panel.cnt[c][mask], panel.observed[mask]) for c in cl.CH5]
    A = [cl.approx_count(z, cap) for z in Z]
    idx = np.flatnonzero(det >= 0)
    out = np.full(idx.size, -1.0)
    for e in np.unique(det[idx]):
        sel = np.flatnonzero(det[idx] == e)
        rows = idx[sel]
        a = int(e) - k + 1
        tot = np.zeros(rows.size)
        ceil_part = np.zeros(rows.size)
        has = np.zeros(rows.size, dtype=bool)
        for z, aa in zip(Z, A):
            R, C, _ = window_change(aa, int(e), k, L, 1)
            d = (R - C)[rows]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base9 = np.nanmean((z[rows][:, a - L : int(e) + 1 - L] == 9).astype(float), axis=1)
            is_ceil = base9 >= 0.5
            tot += d
            ceil_part += np.where(is_ceil, d, 0.0)
            has |= is_ceil
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(tot < 0, ceil_part / tot, np.nan)
        out[sel] = np.where(has, share, -1.0)
    return out
