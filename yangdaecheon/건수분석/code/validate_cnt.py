"""P5 입금 기반 타당성 검증 -- 계획서 `계획/P5_위축기준과_입금검증.md` ⑤-2 ~ ⑤-9 를 그대로 구현한다.

탐지(건수)는 detect_cnt 가 만든다. 이 모듈은 탐지 결과를 입력으로 받아 입금(결과 변수)과 비교만 한다.
월 인덱스는 0 기준(0 = 2023.01), 기준월 e = 계획서의 t - 1 이다. 상대월 j 는 e + j 월이다.

구성
  결과 변수   O1(입금 약화) / O3(잔액 약화) / O2(회복률): g_matrix, recovery_matrix
  탐지 전 값  L_pre(탐지 전 입금 수준), h_pre(탐지 창과 겹치지 않는 입금 추세)
  표본 단위   build_units -> 탐지 기업-월(기업당 1건) + 위험집합 기업-월
  층·표준화   make_cells, eff_cells(공통 지지·업종 병합), estimate(RR = 관찰 / 기대), 기업 이력 군집 부트스트랩
  경로(H-C2)  path_arrays, path_delta, path_summaries
"""
from __future__ import annotations

import math
import warnings

import numpy as np

import cntlib as cl
import detect_cnt as dc

MIN_CTRL = 5            # 공통 지지: 층의 대조군 최소 수
EPS = 1e-9
T = dc.T


# --------------------------------------------------------------------------- 입금·잔액 지표
def _paired(Y: np.ndarray, e: int, lo: int, hi: int, lag: int, need: int):
    """창 [e+lo, e+hi] 와 lag 개월 전 같은 달 창에서 **두 시점 모두 관측된 달만** 짝지어 평균한다. (현재 평균, 비교 평균, 짝 수)."""
    n, T_ = Y.shape
    a, b = e + lo, e + hi
    if a - lag < 0 or b >= T_ or a < 0:
        nan = np.full(n, np.nan)
        return nan, nan.copy(), np.zeros(n, dtype=int)
    cur, old = Y[:, a : b + 1], Y[:, a - lag : b + 1 - lag]
    ok = ~np.isnan(cur) & ~np.isnan(old)
    cnt = ok.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c = np.nanmean(np.where(ok, cur, np.nan), axis=1)
        o = np.nanmean(np.where(ok, old, np.nan), axis=1)
    return np.where(cnt >= need, c, np.nan), np.where(cnt >= need, o, np.nan), cnt


def g_matrix(Y: np.ndarray, e_range, need: int = 5, post=(1, 6)) -> np.ndarray:
    """O1/O3 의 기초: g(i,e) = (사후 6개월 평균 - 같은 달 전년 평균) / 전년 평균. 전년 평균 > 0 이고 짝 >= need 일 때만 정의."""
    n, T_ = Y.shape
    G = np.full((n, T_), np.nan)
    for e in e_range:
        c, o, _ = _paired(Y, e, post[0], post[1], 12, need)
        with np.errstate(divide="ignore", invalid="ignore"):
            G[:, e] = np.where(o > 0, (c - o) / o, np.nan)
    return G


def recovery_matrix(Y: np.ndarray, e_range, need: int = 2) -> np.ndarray:
    """O2: 입금 회복률 = e+4..e+6 평균 / 같은 달 전년 평균 (두 시점 모두 관측된 달만, 전년 평균 > 0)."""
    n, T_ = Y.shape
    R = np.full((n, T_), np.nan)
    for e in e_range:
        c, o, _ = _paired(Y, e, 4, 6, 12, need)
        with np.errstate(divide="ignore", invalid="ignore"):
            R[:, e] = np.where(o > 0, c / o, np.nan)
    return R


def pre_matrices(Y: np.ndarray, k: int, e_range, need: int = 5):
    """탐지 전 입금 수준·추세 (탐지 창 [e-k+1, e] 와 겹치지 않는 구간만).

    L_pre(i,e) = [e-11, e-6] 관측된 달 평균 (O1 의 비교 창과 같은 달).
    h_pre(i,e) = [e-k-5, e-k] 평균을 1년 전 같은 달과 비교한 변화율 (쌍 맞춤, 최소 need 쌍). e >= k+17 에서만 정의.
    """
    n, T_ = Y.shape
    L = np.full((n, T_), np.nan)
    H = np.full((n, T_), np.nan)
    for e in e_range:
        a, b = e - 11, e - 6
        if a >= 0:
            w = Y[:, a : b + 1]
            cnt = (~np.isnan(w)).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmean(w, axis=1)
            L[:, e] = np.where(cnt >= need, m, np.nan)
        c, o, _ = _paired(Y, e, -k - 5, -k, 12, need)
        with np.errstate(divide="ignore", invalid="ignore"):
            H[:, e] = np.where(o > 0, (c - o) / o, np.nan)
    return L, H


def theta_low(G: np.ndarray, status: np.ndarray, e_range, q: float = 5.0) -> float:
    """판정 가능 (i,e) 전체에서 g 의 q번째 백분위수 (고정 문턱: O1 이면 5)."""
    vals = np.concatenate([G[:, e][(status[:, e] == 0) & ~np.isnan(G[:, e])] for e in e_range])
    return float(np.percentile(vals, q))


# --------------------------------------------------------------------------- 규모·업종 층
def size_edges(C: np.ndarray, status: np.ndarray, e_range) -> np.ndarray:
    """규모(탐지 이전 비교 기간 평균 I1) 5분위 경계: 판정 가능 (i,e)의 C 의 20·40·60·80 백분위수."""
    vals = np.concatenate([C[:, e][(status[:, e] == 0) & ~np.isnan(C[:, e])] for e in e_range])
    return np.percentile(vals, [20, 40, 60, 80])


def size_group(C: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = np.searchsorted(edges, C, side="right").astype(np.int8)
    out[np.isnan(C)] = -1
    return out


def sector_groups(sector_all: np.ndarray, sector_p36: np.ndarray, n_levels: int, top: int = 5):
    """업종 대분류 상위 top 개(P36 기업 수 기준)는 그대로, 나머지·결측은 '기타'(= top)."""
    cnt = np.bincount(sector_p36[sector_p36 >= 0], minlength=n_levels)
    top_codes = [int(x) for x in np.argsort(-cnt, kind="stable")[:top]]
    mp = {c: i for i, c in enumerate(top_codes)}
    out = np.array([mp.get(int(s), top) for s in sector_all], dtype=np.int8)
    return out, top_codes


# --------------------------------------------------------------------------- 표본 단위
class Units:
    """(기업, 기준월) 단위: 탐지 기업-월과 위험집합(대조군) 기업-월."""

    def __init__(self, firm, e, is_det, ev, size, sector):
        self.firm, self.e, self.is_det, self.ev, self.size, self.sector = firm, e, is_det, ev, size, sector
        self.x = {}                     # 추가 변수 (균형표용 등)
        self.meta = {}

    def __len__(self):
        return len(self.firm)

    def sub(self, mask):
        u = Units(self.firm[mask], self.e[mask], self.is_det[mask], self.ev[mask], self.size[mask], self.sector[mask])
        u.x = {k: v[mask] for k, v in self.x.items()}
        u.meta = dict(self.meta)
        return u


def build_units(det, status, Gm, theta, size, sector, e_range, cellmask=None):
    """탐지 기업(기준월 = 최초 탐지월)과 위험집합(각 e 에 판정 가능했고 e 까지 한 번도 탐지되지 않은 기업; 이후 탐지돼도 유지).

    결과 g 가 정의되지 않은 칸은 두 집단 모두에서 제외하고 meta 로 센다. cellmask 가 True 인 칸은 판정에서 뺀다(탐지는 호출 쪽에서 다시 계산).
    """
    F, E, D = [], [], []
    miss_det = 0
    for e in e_range:
        judge = status[:, e] == 0
        if cellmask is not None:
            judge = judge & ~cellmask[:, e]
        d_all = judge & (det == e)
        ok = judge & ~np.isnan(Gm[:, e])
        d_ = d_all & ok
        c_ = ok & ((det < 0) | (det > e))
        miss_det += int((d_all & ~ok).sum())
        for mask, flag in ((d_, True), (c_, False)):
            idx = np.flatnonzero(mask)
            F.append(idx)
            E.append(np.full(idx.size, e))
            D.append(np.full(idx.size, flag))
    firm, e, is_det = np.concatenate(F), np.concatenate(E), np.concatenate(D)
    ev = Gm[firm, e] <= theta + EPS
    u = Units(firm, e, is_det, ev, size[firm, e].astype(int), sector[firm].astype(int))
    u.meta = {"탐지 중 결과 미정의로 제외": miss_det}
    return u


# --------------------------------------------------------------------------- 층·공통 지지·표준화
def make_cells(cell_raw: np.ndarray, grp_raw: np.ndarray):
    """원시 층 번호와 병합용 상위 층 번호를 0..C-1 / 0..G-1 로 압축. 반환: (단위별 층 번호, 층별 상위 층 번호)."""
    ucell, cell_u = np.unique(cell_raw, return_inverse=True)
    g_of_cell = np.zeros(len(ucell), dtype=np.int64)
    g_of_cell[cell_u] = grp_raw
    _, cell_grp = np.unique(g_of_cell, return_inverse=True)
    return cell_u, cell_grp


def cells_main(u: Units, n_sector: int = 6):
    """주 층: 기준월 x 규모 5분위 x 업종군, 병합용 상위 층 = 기준월 x 규모 (업종군을 합침)."""
    return make_cells((u.e * 5 + u.size) * n_sector + u.sector, u.e * 5 + u.size)


def _counts(cell_u, C, w, is_det, ev):
    wd, wc = w * is_det, w * (~is_det)
    return (np.bincount(cell_u, wd, C), np.bincount(cell_u, wc, C), np.bincount(cell_u, wd * ev, C), np.bincount(cell_u, wc * ev, C))


def eff_cells(n1, n0, cell_grp, min_ctrl: int = MIN_CTRL):
    """층별 '유효 층' 번호: 탐지군이 있는 층에 대조군이 min_ctrl 미만이면 그 상위 층(업종 병합)으로 올리고, 그래도 대조군이 모자라면 -1(제외).

    상위 층에 하나라도 모자란 하위 층이 있으면 그 상위 층 전체를 병합 층으로 쓴다. 반환: 길이 C 정수 배열 (0..C-1 = 자기 층, C+g = 상위 층 g).
    """
    C = len(n1)
    Gn = int(cell_grp.max()) + 1
    need_fb = (n1 > 0) & (n0 < min_ctrl)
    fb = np.bincount(cell_grp, need_fb.astype(float), Gn) > 0
    gn1 = np.bincount(cell_grp, n1, Gn)
    gn0 = np.bincount(cell_grp, n0, Gn)
    sup_g = fb & (gn1 > 0) & (gn0 >= min_ctrl)
    eff = np.where(fb[cell_grp], np.where(sup_g[cell_grp], C + cell_grp, -1), np.arange(C))
    return eff


def _rr_from_counts(n1, n0, ev1, ev0, cell_grp, min_ctrl: int = MIN_CTRL):
    """RR = 관찰 약화 건수 / 기대 약화 건수 (기대 = 층별 탐지군 수 x 대조군 약화율). 분자·분모가 같은 층·같은 가중."""
    C = len(n1)
    Gn = int(cell_grp.max()) + 1
    need_fb = (n1 > 0) & (n0 < min_ctrl)
    fb = np.bincount(cell_grp, need_fb.astype(float), Gn) > 0
    use = ~fb[cell_grp] & (n1 > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_c = np.where(use & (n0 > 0), n1 * ev0 / n0, 0.0)
    obs, exp, n1s = float(ev1[use].sum()), float(exp_c.sum()), float(n1[use].sum())
    gn1, gn0 = np.bincount(cell_grp, n1, Gn), np.bincount(cell_grp, n0, Gn)
    gev1, gev0 = np.bincount(cell_grp, ev1, Gn), np.bincount(cell_grp, ev0, Gn)
    sup = fb & (gn1 > 0) & (gn0 >= min_ctrl)
    obs += float(gev1[sup].sum())
    exp += float((gn1[sup] * gev0[sup] / gn0[sup]).sum())
    n1s += float(gn1[sup].sum())
    tot = float(n1.sum())
    rr = obs / exp if exp > 0 else math.nan
    return {"rr": rr, "p_det": obs / n1s if n1s > 0 else math.nan, "p_ctrl": exp / n1s if n1s > 0 else math.nan, "obs": obs, "exp": exp,
            "n1": tot, "n1_supported": n1s, "excluded_share": 1 - n1s / tot if tot > 0 else math.nan}


def estimate(u: Units, cell_u, cell_grp, w=None, ev=None, min_ctrl: int = MIN_CTRL):
    """점추정. w: 단위 가중(부트스트랩 중복 횟수), ev: 결과(기본 u.ev)."""
    w = np.ones(len(u)) if w is None else w
    ev = u.ev if ev is None else ev
    n1, n0, ev1, ev0 = _counts(cell_u, int(cell_u.max()) + 1, w, u.is_det, ev.astype(float))
    out = _rr_from_counts(n1, n0, ev1, ev0, cell_grp, min_ctrl)
    out["events_det"] = float(ev1.sum())
    out["n0"] = float(n0.sum())
    return out


def _multiplicity(rng, F):
    return np.bincount(rng.integers(0, F, F), minlength=F).astype(float)


def bootstrap_rr(u: Units, cell_u, cell_grp, F: int, B: int = 2000, seed: int = 20261001, ev=None, min_ctrl: int = MIN_CTRL, keep_nan: bool = False):
    """기업 이력 단위 군집 부트스트랩. 기업(0..F-1)을 복원추출하고 그 기업의 **모든 기준월 기록을 함께** 복제한다.
    매 반복에서 층 집계·공통 지지(업종 병합)·표준화를 다시 계산한다 (탐지 규칙·문턱·규모 경계는 고정).
    반환: (RR 배열, 분모 0 등으로 정의되지 않은 반복 수)."""
    rng = np.random.default_rng(seed)
    ev = (u.ev if ev is None else ev).astype(float)
    C = int(cell_u.max()) + 1
    out = np.empty(B)
    for b in range(B):
        w = _multiplicity(rng, F)[u.firm]
        n1, n0, ev1, ev0 = _counts(cell_u, C, w, u.is_det, ev)
        out[b] = _rr_from_counts(n1, n0, ev1, ev0, cell_grp, min_ctrl)["rr"]
    bad = int(np.isnan(out).sum())
    if keep_nan:
        return out, bad                          # P7: 반복 번호를 유지해 같은 재표집끼리 짝지어 비교할 때
    return out[~np.isnan(out)], bad


def summarize_boot(rr_hat: float, rr_star: np.ndarray, B: int):
    """양측 95% 백분위 CI, 단측 p = (1 + #{RR* <= 1}) / (B' + 1). 채택 = CI 하한 > 1."""
    if rr_star.size == 0:
        return {"rr": rr_hat, "lo": math.nan, "hi": math.nan, "p_one": math.nan, "adopt": False, "B_valid": 0}
    lo, hi = np.percentile(rr_star, [2.5, 97.5])
    p = (1 + int((rr_star <= 1.0).sum())) / (rr_star.size + 1)
    return {"rr": rr_hat, "lo": float(lo), "hi": float(hi), "p_one": float(p), "adopt": bool(lo > 1.0), "B_valid": int(rr_star.size)}


def mean_diff(u: Units, cell_u, cell_grp, outcome: np.ndarray, w=None, min_ctrl: int = MIN_CTRL):
    """연속 결과(O2)의 표준화 평균 차이 = 탐지군 평균 - 표준화 대조군 평균 (ATT 가중, 공통 지지·업종 병합 동일)."""
    w = np.ones(len(u)) if w is None else w
    ok = ~np.isnan(outcome)
    y = np.where(ok, outcome, 0.0)
    C = int(cell_u.max()) + 1
    wd, wc = w * u.is_det * ok, w * (~u.is_det) * ok
    n1, n0 = np.bincount(cell_u, wd, C), np.bincount(cell_u, wc, C)
    s1, s0 = np.bincount(cell_u, wd * y, C), np.bincount(cell_u, wc * y, C)
    Gn = int(cell_grp.max()) + 1
    need_fb = (n1 > 0) & (n0 < min_ctrl)
    fb = np.bincount(cell_grp, need_fb.astype(float), Gn) > 0
    use = ~fb[cell_grp] & (n1 > 0)
    gn1, gn0 = np.bincount(cell_grp, n1, Gn), np.bincount(cell_grp, n0, Gn)
    gs1, gs0 = np.bincount(cell_grp, s1, Gn), np.bincount(cell_grp, s0, Gn)
    sup = fb & (gn1 > 0) & (gn0 >= min_ctrl)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = float(s1[use].sum() + gs1[sup].sum())
        n1s = float(n1[use].sum() + gn1[sup].sum())
        e0 = float(np.where(use & (n0 > 0), n1 * s0 / n0, 0.0).sum() + (gn1[sup] * gs0[sup] / gn0[sup]).sum())
    return {"det_mean": m1 / n1s if n1s > 0 else math.nan, "ctrl_std_mean": e0 / n1s if n1s > 0 else math.nan,
            "diff": (m1 - e0) / n1s if n1s > 0 else math.nan}


def bootstrap_mean_diff(u: Units, cell_u, cell_grp, outcome, F: int, B: int = 1000, seed: int = 20261002):
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        out[b] = mean_diff(u, cell_u, cell_grp, outcome, _multiplicity(rng, F)[u.firm])["diff"]
    out = out[~np.isnan(out)]
    return out


# --------------------------------------------------------------------------- 균형표 (표준화 차이)
def std_diff(x: np.ndarray, u: Units, cell_u, cell_grp, min_ctrl: int = MIN_CTRL):
    """변수 x 의 (탐지군 평균, 대조군 평균(조정 전), 대조군 표준화 평균, 표준화 차이 조정 전/후). x 가 NaN 인 단위는 뺀다."""
    ok = ~np.isnan(x)
    y = np.where(ok, x, 0.0)
    C = int(cell_u.max()) + 1
    wd, wc = (u.is_det & ok).astype(float), ((~u.is_det) & ok).astype(float)
    n1, n0 = np.bincount(cell_u, wd, C), np.bincount(cell_u, wc, C)
    s1, s0 = np.bincount(cell_u, wd * y, C), np.bincount(cell_u, wc * y, C)
    eff = eff_cells(n1, n0, cell_grp, min_ctrl)
    Gn = int(cell_grp.max()) + 1
    E = C + Gn
    m = (eff[cell_u] >= 0)
    e_u = eff[cell_u]
    N1 = np.bincount(e_u[m], wd[m], E)
    N0 = np.bincount(e_u[m], wc[m], E)
    S1 = np.bincount(e_u[m], (wd * y)[m], E)
    S0 = np.bincount(e_u[m], (wc * y)[m], E)
    sup = (N1 > 0) & (N0 > 0)
    tot1 = N1[sup].sum()
    mean1 = S1[sup].sum() / tot1 if tot1 > 0 else math.nan
    mean0_std = float((N1[sup] * S0[sup] / N0[sup]).sum() / tot1) if tot1 > 0 else math.nan
    x1, x0 = x[u.is_det & ok], x[(~u.is_det) & ok]
    mean0_raw = float(x0.mean()) if x0.size else math.nan
    v = 0.5 * (x1.var() + x0.var()) if x1.size and x0.size else math.nan
    sd = math.sqrt(v) if v and v > 0 else math.nan
    return {"det": float(x1.mean()) if x1.size else math.nan, "ctrl_raw": mean0_raw, "ctrl_std": mean0_std,
            "smd_raw": (float(x1.mean()) - mean0_raw) / sd if sd == sd else math.nan, "smd_adj": (mean1 - mean0_std) / sd if sd == sd else math.nan}


# --------------------------------------------------------------------------- H-C2: 입금 경로
def path_arrays(Yw: np.ndarray, firm: np.ndarray, e: np.ndarray, J):
    """단위별 상대월 j 의 (입금, 전년 같은 달 입금, 둘 다 관측 여부)."""
    T_ = Yw.shape[1]
    m = len(firm)
    num = np.zeros((m, len(J)))
    den = np.zeros((m, len(J)))
    valid = np.zeros((m, len(J)), dtype=bool)
    for jj, j in enumerate(J):
        a, b = e + j, e + j - 12
        inr = (a >= 0) & (a < T_) & (b >= 0)
        cur = Yw[firm, np.clip(a, 0, T_ - 1)]
        old = Yw[firm, np.clip(b, 0, T_ - 1)]
        ok = inr & ~np.isnan(cur) & ~np.isnan(old)
        num[:, jj], den[:, jj], valid[:, jj] = np.where(ok, cur, 0.0), np.where(ok, old, 0.0), ok
    return num, den, valid


def path_delta(num, den, valid, eff_u, is_det, w):
    """층·집단별 입금 합의 전년 대비 비 R_s^G(j) = sum y(e+j) / sum y(e+j-12) 를 탐지군 층 가중으로 표준화한 뒤 차이 Δ(j) = R^1 - R^0.
    분모가 0 이거나 한쪽 집단이 없는 층·j 는 제외하고 가중치를 다시 정규화한다. 반환 (Δ, R1, R0, 사용 층 수)."""
    Jn = num.shape[1]
    m = eff_u >= 0
    E = int(eff_u.max()) + 1 if m.any() else 1
    out = np.full((4, Jn), np.nan)
    for jj in range(Jn):
        v = m & valid[:, jj]
        if not v.any():
            continue
        wd, wc = w * is_det * v, w * (~is_det) * v
        nu1 = np.bincount(eff_u[v], (wd * num[:, jj])[v], E)
        de1 = np.bincount(eff_u[v], (wd * den[:, jj])[v], E)
        nu0 = np.bincount(eff_u[v], (wc * num[:, jj])[v], E)
        de0 = np.bincount(eff_u[v], (wc * den[:, jj])[v], E)
        w1 = np.bincount(eff_u[v], wd[v], E)
        sup = (de1 > 0) & (de0 > 0) & (w1 > 0)
        if not sup.any():
            continue
        ws = w1[sup] / w1[sup].sum()
        r1, r0 = float((ws * nu1[sup] / de1[sup]).sum()), float((ws * nu0[sup] / de0[sup]).sum())
        out[:, jj] = (r1 - r0, r1, r0, int(sup.sum()))
    return out[0], out[1], out[2], out[3]


def path_summaries(delta, J):
    """건수 감소 이전(j <= -6)·동시(-5..0)·사후(1..6) 평균 Δ 와 사전 기울기 (판정 문턱 없음)."""
    J = np.asarray(J)
    pre, conc, post = (J >= -12) & (J <= -6), (J >= -5) & (J <= 0), (J >= 1)

    def mean(m):
        v = delta[m]
        return float(np.nanmean(v)) if np.isfinite(v).any() else math.nan

    x, y = J[pre].astype(float), delta[pre]
    ok = np.isfinite(y)
    slope = math.nan
    if ok.sum() >= 3:
        xx = x[ok] - x[ok].mean()
        slope = float((xx * (y[ok] - y[ok].mean())).sum() / (xx ** 2).sum())
    return {"pre_mean": mean(pre), "slope_pre": slope, "conc_mean": mean(conc), "post_mean": mean(post)}


def bootstrap_path(u: Units, cell_u, cell_grp, num, den, valid, J, F: int, B: int = 1000, seed: int = 20261003, min_ctrl: int = MIN_CTRL):
    """H-C2 경로 요약의 군집 부트스트랩 (매 반복에서 공통 지지·층 병합을 다시 계산)."""
    rng = np.random.default_rng(seed)
    C = int(cell_u.max()) + 1
    keys = ["pre_mean", "slope_pre", "conc_mean", "post_mean"]
    res = {k: [] for k in keys}
    deltas = []
    for b in range(B):
        w = _multiplicity(rng, F)[u.firm]
        n1, n0, _, _ = _counts(cell_u, C, w, u.is_det, np.zeros(len(u)))
        eff = eff_cells(n1, n0, cell_grp, min_ctrl)
        d, _, _, _ = path_delta(num, den, valid, eff[cell_u], u.is_det, w)
        deltas.append(d)
        s = path_summaries(d, J)
        for k in keys:
            res[k].append(s[k])
    return {k: np.array(v) for k, v in res.items()}, np.array(deltas)


def ci(arr, lo=2.5, hi=97.5):
    a = arr[np.isfinite(arr)]
    return (float(np.percentile(a, lo)), float(np.percentile(a, hi))) if a.size else (math.nan, math.nan)


def interpret_pre(pre_ci, slope_ci):
    """H-C2 해석 문구(판정 문턱 없음, '선행'·'조기경보' 표현 금지). 건수 감소 이전(j <= -6)의 평균 Δ 와 사전 기울기의 95% CI 로 방향까지 구분한다.
    Δ = 탐지군 - 대조군 이므로 음수면 탐지군의 입금이 더 약하다는 뜻이다."""
    inc0 = lambda c: c[0] <= 0 <= c[1]
    if inc0(pre_ci) and inc0(slope_ci):
        return "건수 감소 이전(j ≤ −6)에 뚜렷한 입금 차이가 확인되지 않음"
    if not inc0(pre_ci):
        if pre_ci[1] < 0:
            return "건수 감소 이전부터 탐지군의 입금이 대조군보다 이미 약했음 → 건수가 먼저 움직였다는 근거 없음"
        return ("건수 감소 이전에는 탐지군의 입금이 대조군보다 오히려 높았음 (평균 회귀 가능성: 탐지 직전의 일시적 호조 뒤 하락) "
                "→ 건수가 입금보다 먼저 움직였다고 해석하지 않음")
    return "건수 감소 이전 구간에서 입금 차이가 추세적으로 변하고 있음(사전 기울기 ≠ 0) → 건수가 먼저 움직였다는 근거 없음"


# --------------------------------------------------------------------------- 표본 준비 (36개월·35개월 법인이 같은 코드를 쓰도록)
def first_detect_masked(ch, phi: float, thr: float, cellmask: np.ndarray, e_lo: int | None = None) -> np.ndarray:
    """dc.first_detect 와 같되 cellmask 가 True 인 칸은 탐지 후보에서 뺀다 (S-C1: 천장 채널이 있는 칸 제외)."""
    lo = dc.e_lo_for(ch.k, ch.L, e_lo)
    r, C = ch.r[:, lo : dc.E_MAX + 1], ch.C[:, lo : dc.E_MAX + 1]
    with np.errstate(invalid="ignore"):
        cond = ~np.isnan(r) & (C >= phi) & (r <= -thr + EPS) & ~cellmask[:, lo : dc.E_MAX + 1]
    has = cond.any(axis=1)
    return np.where(has, cond.argmax(axis=1) + lo, -1)


def fit_constants(panel: cl.CountPanel, mask: np.ndarray, mode: str, K: int, phi: float, n_pct: float, I: np.ndarray = None) -> dict:
    """**P36 에서 한 번 정해 고정하는 상수**: θ_O1·θ_O3, 규모 경계, 업종군 상위 5개, 입금 윈저화 상한, θ_h·L_pre/h_pre 3분위 경계."""
    I = dc.i1_matrix(panel, mask) if I is None else I          # P7: 다른 활동량 지수(I2 등)를 줄 수 있다
    res = dc.apply_rule(I, mode, K, phi, n_pct)
    st, ch = res["status"], res["change"]
    er = list(dc.e_range(K))
    Y = panel.amt["요구불입금금액"][mask]
    Bal = panel.amt["요구불예금잔액"][mask]
    G, Gb = g_matrix(Y, er), g_matrix(Bal, er)
    R2 = recovery_matrix(Y, er)
    L, H = pre_matrices(Y, K, er)
    e_h = [e for e in er if e >= K + 17]
    hv = np.concatenate([H[:, e][(st[:, e] == 0) & ~np.isnan(H[:, e]) & ~np.isnan(L[:, e])] for e in e_h])
    lv = np.concatenate([L[:, e][(st[:, e] == 0) & ~np.isnan(H[:, e]) & ~np.isnan(L[:, e])] for e in e_h])
    sec = dc.firm_attribute(panel, mask, "업종_대분류")
    secg, top_codes = sector_groups(sec, sec, len(panel.attr_levels["업종_대분류"]))
    r2v = np.concatenate([R2[:, e][(st[:, e] == 0) & ~np.isnan(R2[:, e])] for e in er])
    return {"theta_o1": theta_low(G, st, er), "theta_o3": theta_low(Gb, st, er), "edges": size_edges(ch.C, st, er), "cap_y": float(np.nanpercentile(Y, 99.5)),
            "top_codes": top_codes, "theta_h": float(np.percentile(hv, 5)), "l_terc": np.percentile(lv, [100 / 3, 200 / 3]), "h_terc": np.percentile(hv, [100 / 3, 200 / 3]),
            "o2_cap": float(np.percentile(r2v, 99)), "mode": mode, "K": K, "phi": phi, "n_pct": n_pct}


def prepare(panel: cl.CountPanel, mask: np.ndarray, consts: dict, phi: float = None, n_pct: float = None, drop_ceiling_cells: bool = False,
            cap: float = cl.DEFAULT_CAP, I: np.ndarray = None, post=(1, 6), theta_q: float = None, e_lo: int | None = None) -> dict:
    """표본 하나의 분석 재료를 만든다: 탐지(고정 규칙), 결과 g, 규모·업종 층, 탐지군·위험집합 단위, 탐지 전 입금 수준·추세.

    phi, n_pct 를 주면 그 규칙(예: v1)으로 탐지한다. drop_ceiling_cells=True 이면 S-C1: 천장 채널이 있는 칸을 탐지 후보·위험집합에서 뺀다.
    문턱(θ)·경계는 모두 consts(P36 에서 고정)를 쓴다. e_lo: 첫 후보월(기본 k+11 = 계획서 범위, P10 후보월 확장에서만 바꿈)."""
    mode, K = consts["mode"], consts["K"]
    phi = consts["phi"] if phi is None else phi
    n_pct = consts["n_pct"] if n_pct is None else n_pct
    L_ = dc.lag_for(mode, K)
    er = list(dc.e_range(K, e_lo))
    I = dc.i1_matrix(panel, mask, cap) if I is None else I
    res = dc.apply_rule(I, mode, K, phi, n_pct, e_lo)
    st, ch, det = res["status"], res["change"], res["det"]
    FL = dc.ceiling_window_flag(panel, mask, K, L_)
    cellmask = None
    if drop_ceiling_cells:
        det = first_detect_masked(ch, phi, n_pct / 100.0, FL, e_lo)
        cellmask = FL
    Y = panel.amt["요구불입금금액"][mask]
    Bal = panel.amt["요구불예금잔액"][mask]
    need = (post[1] - post[0] + 1) - 1                 # 6개월 추적이면 5쌍, 3개월 추적이면 2쌍
    G, Gb, R2 = g_matrix(Y, er, need, post), g_matrix(Bal, er, need, post), recovery_matrix(Y, er)
    Lm, Hm = pre_matrices(Y, K, er)
    size = size_group(ch.C, consts["edges"])
    sec = dc.firm_attribute(panel, mask, "업종_대분류")
    mp = {c: i for i, c in enumerate(consts["top_codes"])}
    secg = np.array([mp.get(int(s), len(consts["top_codes"])) for s in sec], dtype=np.int8)
    if theta_q is None and tuple(post) == (1, 6):
        theta = consts["theta_o1"]
    else:                                              # P7: 입금 약화 정의를 바꾼 변형은 문턱을 P36 판정 가능 전체에서 같은 방식으로 다시 구한다
        theta = theta_low(G, st, er, 5.0 if theta_q is None else theta_q)
    U = build_units(det, st, G, theta, size, secg, er, cellmask)
    U.x["L_pre"], U.x["h_pre"] = Lm[U.firm, U.e], Hm[U.firm, U.e]
    U.x["o3"] = (Gb[U.firm, U.e] <= consts["theta_o3"] + EPS).astype(float)
    U.x["o3_def"] = ~np.isnan(Gb[U.firm, U.e])
    U.x["o2"] = R2[U.firm, U.e]
    cu, cg = cells_main(U)
    return {"er": er, "I": I, "res": res, "det": det, "status": st, "change": ch, "FL": FL, "Y": Y, "Bal": Bal, "G": G, "Gb": Gb, "R2": R2, "L": Lm, "H": Hm,
            "size": size, "sector": secg, "units": U, "cell_u": cu, "cell_grp": cg, "F": int(mask.sum()), "phi": phi, "n_pct": n_pct,
            "Yw": np.minimum(Y, consts["cap_y"]), "theta": theta}


def analyze_units(u: Units, F: int, B: int = 2000, seed: int = 20261001, cells=None, ev=None, keep_all: bool = False) -> dict:
    """RR 점추정 + 군집 부트스트랩 CI·p·채택 + 지원 정보. 탐지군이 없으면 NaN."""
    if int(u.is_det.sum()) == 0 or int((~u.is_det).sum()) == 0:
        return {"rr": math.nan, "lo": math.nan, "hi": math.nan, "p_one": math.nan, "adopt": False, "B_valid": 0, "B_bad": 0, "n1": 0.0, "n1_supported": 0.0,
                "excluded_share": math.nan, "p_det": math.nan, "p_ctrl": math.nan, "obs": 0.0, "exp": 0.0, "events_det": 0.0, "n0": 0.0}
    cu, cg = cells if cells is not None else cells_main(u)
    est = estimate(u, cu, cg, ev=ev)
    star_all, bad = bootstrap_rr(u, cu, cg, F, B, seed, ev=ev, keep_nan=True)
    star = star_all[~np.isnan(star_all)]
    s = summarize_boot(est["rr"], star, B)
    s.update({k: est[k] for k in ("p_det", "p_ctrl", "n1", "n1_supported", "excluded_share", "obs", "exp", "events_det", "n0")})
    s["B_bad"] = bad
    s["star"] = star
    if keep_all:
        s["star_all"] = star_all                       # P7: 반복 번호를 유지한 부트스트랩 (짝지은 비교용)
    return s


def cells_s1(u: Units, consts: dict):
    """S1(입금 수준·추세 직접 조정): 기준월 {e 23–25, 26–29} x 규모 5분위 x L_pre 3분위 x h_pre 3분위. 병합 없음(층이 모자라면 제외).
    h_pre·L_pre 가 정의되지 않은 단위는 호출 쪽에서 미리 뺀다."""
    blk = (u.e >= 26).astype(int)
    lq = np.searchsorted(consts["l_terc"], u.x["L_pre"], side="right")
    hq = np.searchsorted(consts["h_terc"], u.x["h_pre"], side="right")
    raw = ((blk * 5 + u.size) * 3 + lq) * 3 + hq
    return make_cells(raw, raw)


def path_median_delta(num, den, valid, eff_u, is_det):
    """민감도: 층 안에서 기업별 변화율(y(e+j)/y(e+j-12) - 1, 전년 입금 > 0)의 중앙값을 탐지군 층 가중으로 표준화한 차이 (점추정만)."""
    Jn = num.shape[1]
    out = np.full(Jn, np.nan)
    for jj in range(Jn):
        v = (eff_u >= 0) & valid[:, jj] & (den[:, jj] > 0)
        if not v.any():
            continue
        r = num[v, jj] / den[v, jj] - 1.0
        e_, d_ = eff_u[v], is_det[v]
        num_w = tot_w = 0.0
        for s_ in np.unique(e_[d_]):
            a, b = r[(e_ == s_) & d_], r[(e_ == s_) & ~d_]
            if a.size and b.size:
                num_w += a.size * (np.median(a) - np.median(b))
                tot_w += a.size
        out[jj] = num_w / tot_w if tot_w > 0 else np.nan
    return out
