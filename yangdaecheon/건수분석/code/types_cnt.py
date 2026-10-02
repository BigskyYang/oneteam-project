"""P6 어떤 기업에서 위축 신호가 더 강한가 -- 계획서 `계획/P6_채널유형화_위축비교.md` ⑥-0 ~ §3 을 그대로 구현한다.

탐지(건수)는 detect_cnt, 입금 결과·층·RR 은 validate_cnt 를 그대로 재사용한다. 이 모듈은 그 위에
  유형 정의   : 2023.01~06 프로파일의 오프라인 비중 3분위, 자동이체 높음/낮음 (경계는 P36 에서 고정해 P35만에 그대로 적용)
  C1 탐지 비율: 유형 x 규모 5분위 탐지율과 직접 표준화(규모 분포 고정 가중) 탐지 비율
  C2·C3 배수  : 유형(자동이체 수준)별 RR -- **같은 집단 안의 비탐지 기업**과만 비교, 같은 기업 재표집으로 두 배수의 비
  C4 이동     : 총 활동은 유지한 채 오프라인 비중이 줄어든 '채널 구성 이동' 신호 (위축 탐지와 별개의 신호)
를 더한다. 월 인덱스 0 = 2023.01, 기준월 e. 실제 데이터를 쓰지 않는 검증은 tools/test_types_cnt.py.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

import cntlib as cl
import detect_cnt as dc
import validate_cnt as vc

PROFILE = (0, 5)          # 프로파일 기간 2023.01~06 (가장 이른 기준월 2024.06 의 비교 기준 구간 [e-11, e-6] = 2023.07~ 과 겹치지 않음)
MIN_OBS = 4               # 프로파일 기간 중 관측된 달 >= 4 이어야 유형 정의
SHIFT_DELTA = -0.20       # 오프라인 비중이 6개월 사이 0.20 이상 감소
SHIFT_R = 0.20            # 총 활동(I1) 변화율 +-20% 이내
TYPE_NAMES = ["인터넷·스마트 중심형", "혼합형", "창구·ATM 중심형"]
AUTO_NAMES = ["자동이체 낮음", "자동이체 높음"]
TOL = 1e-12


# --------------------------------------------------------------------------- 유형 정의
def profile(panel: cl.CountPanel, mask: np.ndarray, cap: float = cl.DEFAULT_CAP, months=PROFILE):
    """프로파일 기간의 채널별 평균 근사 건수(관측된 달만), 자동이체 평균 코드, 관측된 달 수. 관측 달 < MIN_OBS 이면 NaN."""
    a, b = months
    obs = panel.observed[mask]
    nobs = obs[:, a : b + 1].sum(axis=1)
    means = {}
    for c in cl.CH5:
        v = cl.approx_count(cl.codes_float(panel.cnt[c][mask], obs), cap)[:, a : b + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(v, axis=1)
        means[c] = np.where(nobs >= MIN_OBS, m, np.nan)
    z = cl.codes_float(panel.cnt[cl.AUTO][mask], obs)[:, a : b + 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        auto = np.nanmean(z, axis=1)
    return means, np.where(nobs >= MIN_OBS, auto, np.nan), nobs


def offline_share(means: dict) -> np.ndarray:
    """오프라인 비중 = (창구 + ATM) / 국내 5채널 합. 합이 0 이면 NaN (유형 미정)."""
    off = means[cl.OFFLINE[0]] + means[cl.OFFLINE[1]]
    tot = sum(means[c] for c in cl.CH5)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, off / tot, np.nan)


def fit_type_edges(share: np.ndarray) -> np.ndarray:
    v = share[np.isfinite(share)]
    return np.percentile(v, [100 / 3, 200 / 3])


def assign_type(share: np.ndarray, edges) -> np.ndarray:
    """경계값과 같으면 **낮은 쪽**: <= 경계1 → 0(인터넷·스마트 중심), <= 경계2 → 1(혼합), 그 위 → 2(창구·ATM 중심). 미정 -1."""
    out = np.full(share.shape, -1, dtype=np.int8)
    ok = np.isfinite(share)
    s = share[ok]
    out[ok] = np.where(s <= edges[0] + TOL, 0, np.where(s <= edges[1] + TOL, 1, 2))
    return out


def fit_auto_edge(auto: np.ndarray) -> float:
    return float(np.median(auto[np.isfinite(auto)]))


def assign_auto(auto: np.ndarray, edge: float) -> np.ndarray:
    """중앙값 이상 = 1(높음, 같으면 높음), 미만 = 0(낮음), 미정 -1."""
    out = np.full(auto.shape, -1, dtype=np.int8)
    ok = np.isfinite(auto)
    out[ok] = (auto[ok] >= edge - TOL).astype(np.int8)
    return out


def make_types(panel, mask, type_edges=None, auto_edge=None, cap: float = cl.DEFAULT_CAP) -> dict:
    """유형·자동이체 수준. 경계를 주지 않으면 이 표본에서 정한다(P36). 주면 그대로 쓴다(P35만)."""
    means, auto, nobs = profile(panel, mask, cap)
    share = offline_share(means)
    te = fit_type_edges(share) if type_edges is None else np.asarray(type_edges, dtype=float)
    ae = fit_auto_edge(auto) if auto_edge is None else float(auto_edge)
    return {"share": share, "auto": auto, "nobs": nobs, "type_edges": te, "auto_edge": ae,
            "type": assign_type(share, te), "auto_level": assign_auto(auto, ae)}


def type_change_rate(panel, mask, cap: float) -> float:
    """cap 을 바꿔 유형을 처음부터 다시 만들면 몇 %의 기업이 다른 유형이 되는가 (두 방식 모두 유형이 정의된 기업 기준)."""
    base, alt = make_types(panel, mask)["type"], make_types(panel, mask, cap=cap)["type"]
    both = (base >= 0) & (alt >= 0)
    return float((base[both] != alt[both]).mean()) if both.any() else math.nan


# --------------------------------------------------------------------------- 공통: 재표집 보조
def ratio_ci(boot: np.ndarray, a: int, b: int, lo: float = 2.5, hi: float = 97.5):
    """부트스트랩 배열(B, G) 에서 열 a / 열 b 의 비와 95% 범위, 유효 반복 수. 한쪽이 정의되지 않았거나 분모가 0 이면 그 반복은 뺀다."""
    x, y = boot[:, a], boot[:, b]
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if ok.sum() < 20:
        return {"lo": math.nan, "hi": math.nan, "valid": int(ok.sum())}
    r = x[ok] / y[ok]
    return {"lo": float(np.percentile(r, lo)), "hi": float(np.percentile(r, hi)), "valid": int(ok.sum())}


def point_ratio(point_a, point_b) -> float:
    return point_a / point_b if (point_a is not None and point_b is not None and np.isfinite(point_a) and np.isfinite(point_b) and point_b > 0) else math.nan


# --------------------------------------------------------------------------- C1: 탐지 비율 (규모를 맞춘 직접 표준화)
def firm_detection_info(D: dict):
    """판정 가능 기업(기준월 중 한 번이라도 판정 가능), 첫 판정 가능월의 규모 5분위, 최초 탐지 여부."""
    ER = D["er"]
    st = D["status"][:, ER]
    jf = (st == 0).any(axis=1)
    first = np.where(jf, (st == 0).argmax(axis=1), 0)
    size = np.where(jf, D["size"][:, ER][np.arange(len(first)), first], -1).astype(int)
    return jf, size, D["det"] >= 0


def _counts_ts(tp, size, det, jf, w, T, Q):
    m = jf & (tp >= 0) & (size >= 0)
    idx = tp[m].astype(int) * Q + size[m]
    ww = np.ones(int(m.sum())) if w is None else w[m]
    n = np.bincount(idx, ww, T * Q).reshape(T, Q)
    d = np.bincount(idx, ww * det[m], T * Q).reshape(T, Q)
    return n, d


def std_support(tp, size, det, jf, T=3, Q=5, min_n=10):
    """직접 표준화의 표준 인구와 지원 범위: 유형이 존재하는 모든 유형에서 기업 수가 min_n 이상인 규모 분위만 쓰고,
    가중치는 그 분위들에서 유형이 정의된 판정 가능 기업 전체의 규모 분포(고정)."""
    n, d = _counts_ts(tp, size, det, jf, None, T, Q)
    present = [t for t in range(T) if n[t].sum() > 0]
    Qs = [q for q in range(Q) if present and all(n[t, q] >= min_n for t in present)]
    tot = n.sum(axis=0)[Qs] if Qs else np.array([])
    wq = tot / tot.sum() if Qs and tot.sum() > 0 else np.array([])
    return {"present": present, "Qs": Qs, "wq": wq, "n": n, "d": d}


def std_rates(tp, size, det, jf, sup, w=None, T=3, Q=5):
    n, d = _counts_ts(tp, size, det, jf, w, T, Q)
    out = np.full(T, np.nan)
    if not sup["Qs"]:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n > 0, d / n, np.nan)
    for t in sup["present"]:
        v = r[t, sup["Qs"]]
        if np.isfinite(v).all():
            out[t] = float((sup["wq"] * v).sum())
    return out


def boot_std_rates(tp, size, det, jf, B: int = 2000, seed: int = 20261101, min_n: int = 10):
    """규모를 맞춘 탐지 비율의 기업 단위 부트스트랩. 반환: 점추정(T,), 부트스트랩 (B, T), 지원 정보."""
    F = len(tp)
    sup = std_support(tp, size, det, jf, min_n=min_n)
    point = std_rates(tp, size, det, jf, sup)
    rng = np.random.default_rng(seed)
    boot = np.full((B, 3), np.nan)
    for b in range(B):
        boot[b] = std_rates(tp, size, det, jf, sup, w=vc._multiplicity(rng, F))
    return {"point": point, "boot": boot, "support": sup}


# --------------------------------------------------------------------------- C2·C3: 같은 집단 안 비교로 구한 배수와 두 배수의 비
def rr_by_group(U: vc.Units, grp_u: np.ndarray, levels, F: int, B: int = 2000, seed: int = 20261102, min_ctrl: int = vc.MIN_CTRL):
    """집단(유형·자동이체 수준)별 RR. 각 집단의 탐지 기업은 **같은 집단의 비탐지 기업**(같은 기준월·규모·업종군, 공통 지지 동일)과만 비교한다.
    같은 기업 재표집(기업의 모든 월 기록을 함께)으로 매 반복에서 위험집합·집단별 RR 을 다시 계산해 (B, G) 배열을 반환한다(정의되지 않은 반복은 NaN)."""
    subs, cells, point = [], [], []
    for g in levels:
        s = U.sub(grp_u == g)
        subs.append(s)
        if int(s.is_det.sum()) == 0 or int((~s.is_det).sum()) == 0:
            cells.append(None)
            point.append(None)
            continue
        cu, cg = vc.cells_main(s)
        cells.append((cu, cg))
        point.append(vc.estimate(s, cu, cg))
    rng = np.random.default_rng(seed)
    boot = np.full((B, len(levels)), np.nan)
    for b in range(B):
        wf = vc._multiplicity(rng, F)
        for gi, s in enumerate(subs):
            if cells[gi] is None:
                continue
            cu, cg = cells[gi]
            n1, n0, ev1, ev0 = vc._counts(cu, int(cu.max()) + 1, wf[s.firm], s.is_det, s.ev.astype(float))
            boot[b, gi] = vc._rr_from_counts(n1, n0, ev1, ev0, cg, min_ctrl)["rr"]
    return {"levels": list(levels), "point": point, "boot": boot, "units": [int(s.is_det.sum()) for s in subs]}


def group_ci(res: dict, gi: int, lo: float = 2.5, hi: float = 97.5):
    v = res["boot"][:, gi]
    v = v[np.isfinite(v)]
    if v.size < 20:
        return math.nan, math.nan, int(v.size)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi)), int(v.size)


# --------------------------------------------------------------------------- C4: 채널 구성 이동 (위축 탐지와 별개의 신호)
def offline_delta(panel, mask, ER, k: int = 6, cap: float = cl.DEFAULT_CAP, need: int = 5) -> np.ndarray:
    """Delta(i,e) = (최근 6개월 [e-5,e] 오프라인 비중) - (직전 6개월 [e-11,e-6] 오프라인 비중). 쌍 맞춤(두 창의 같은 위치 달이 모두 관측된 짝 >= need)."""
    obs = panel.observed[mask]
    Ys = {c: cl.approx_count(cl.codes_float(panel.cnt[c][mask], obs), cap) for c in cl.CH5}
    n, T_ = obs.shape
    Dl = np.full((n, T_), np.nan)
    for e in ER:
        cur, old = {}, {}
        for c in cl.CH5:
            cur[c], old[c], _ = vc._paired(Ys[c], e, -(k - 1), 0, k, need)
        oc = sum(cur[c] for c in cl.OFFLINE)
        tc = sum(cur.values())
        oo = sum(old[c] for c in cl.OFFLINE)
        to = sum(old.values())
        with np.errstate(divide="ignore", invalid="ignore"):
            sc = np.where(tc > 0, oc / tc, np.nan)
            so = np.where(to > 0, oo / to, np.nan)
        Dl[:, e] = sc - so
    return Dl


def build_shift_units(D: dict, delta: np.ndarray, theta: float, activity_kept: bool = True,
                      d_thr: float = SHIFT_DELTA, r_thr: float = SHIFT_R, n_pct: float = None):
    """이동 기업-월(기업당 최초 이동월)과 위험집합(그 달 판정 가능했고 이동·위축 탐지를 아직 겪지 않은 기업).
    기준월까지 끝난 창만 쓴다. activity_kept=True: Delta <= -0.20 이고 |I1 변화율| <= 20% (총 활동 유지).
    activity_kept=False: 총 활동 조건을 빼고 위축 탐지 문턱(-n%)에 못 미치는 칸만 (조건화의 영향 점검). 이미 위축 탐지된 달 이후는 모두 뺀다."""
    st, ch, det, ER = D["status"], D["change"], D["det"], D["er"]
    n, T_ = st.shape
    already = (det[:, None] >= 0) & (np.arange(T_)[None, :] >= det[:, None])
    npct = (D["n_pct"] if n_pct is None else n_pct) / 100.0
    with np.errstate(invalid="ignore"):
        act = (np.abs(ch.r) <= r_thr + 1e-9) if activity_kept else (ch.r > -npct + 1e-9)
        cond = (st == 0) & ~already & (delta <= d_thr + 1e-9) & act
    in_er = np.zeros(T_, dtype=bool)
    in_er[list(ER)] = True
    cond &= in_er[None, :]
    has = cond.any(axis=1)
    first = np.where(has, cond.argmax(axis=1), -1)
    U = vc.build_units(first, st, D["G"], theta, D["size"], D["sector"], ER, cellmask=already)
    return U, first


def analyze_shift(D: dict, delta: np.ndarray, consts: dict, F: int, B: int = 2000, seed: int = 20261103,
                  activity_kept: bool = True, min_units: int = 30, min_events: int = 5) -> dict:
    """이동 기업의 입금 약화 배수. 이동 기업 < min_units 이거나 약화 사건 < min_events 이면 배수를 계산하지 않고 기술만."""
    U, first = build_shift_units(D, delta, consts["theta_o1"], activity_kept)
    n1 = int(U.is_det.sum())
    ev = int((U.ev & U.is_det).sum())
    out = {"n_units": n1, "events": ev, "ctrl_units": int((~U.is_det).sum()), "descriptive": n1 < min_units or ev < min_events,
           "size_dist": [int(((U.size == q) & U.is_det).sum()) for q in range(5)], "units": U}
    if not out["descriptive"]:
        out["R"] = vc.analyze_units(U, F, B, seed)
    return out


# --------------------------------------------------------------------------- 한 표본의 비교 전체 (P36 과 P35만이 같은 코드를 쓰도록)
def run_all(panel, mask, D: dict, T: dict, consts: dict, B: int = 2000, seed0: int = 20261100, min_det: int = 30) -> dict:
    """C1~C4 를 한 표본에서 계산한다. D = vc.prepare 결과, T = make_types 결과. 유형 경계·자동이체 경계는 T 가 가진 값(P36 에서 고정)."""
    F = D["F"]
    U = D["units"]
    jf, size_f, det_f = firm_detection_info(D)
    res = {"jf": jf, "size_f": size_f, "det_f": det_f}
    res["c1"] = boot_std_rates(T["type"], size_f, det_f, jf, B=B, seed=seed0 + 1)
    typ_u = T["type"][U.firm]
    aut_u = T["auto_level"][U.firm]
    res["c2"] = rr_by_group(U, typ_u, [0, 1, 2], F, B=B, seed=seed0 + 2)
    res["c3"] = rr_by_group(U, aut_u, [0, 1], F, B=B, seed=seed0 + 3)
    delta = offline_delta(panel, mask, D["er"])
    res["delta"] = delta
    res["c4_kept"] = analyze_shift(D, delta, consts, F, B=B, seed=seed0 + 4, activity_kept=True)
    res["c4_any"] = analyze_shift(D, delta, consts, F, B=B, seed=seed0 + 5, activity_kept=False)
    return res
