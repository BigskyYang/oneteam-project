"""P10 추가 민감도 -- 계획서 `계획/P10_추가민감도_후보월확장과_대칭결과.md`.

본 분석(P5)의 두 가지 설계 비대칭을 점검한다. 신호 규칙(비교 기준·k·Φ·n)과 요구불입금액 약화 문턱·규모 경계·업종군은 **P36 에서 정한 값을 그대로 고정**하고 한 축만 바꾼다.

  A  후보월 확장 : 계획서가 전년 비교(L=12)를 가정해 정한 후보월 범위(2024.06~2025.06, e=17..29)를 직전 기간 비교(L=6)가 허용하는 가장 이른 달
                   (e=11, 2023.12)까지 넓힌다. 신호 후 6개월 추적 때문에 끝(2025.06)은 그대로다.
  B  대칭 결과  : 요구불입금액 약화를 '신호 후 6개월 평균이 **직전** 구간 평균 대비 얼마나 변했나'로 다시 정의한다(건수 신호가 직전 기간 대비인 것과 같은 종류).
                   B1 직전 6개월 [e-5, e], B2 직전 12개월 [e-11, e]. 이 구간은 신호의 비교 달 [e-11, e-6] 과 겹치지 않거나(B1) 결과를 가리는 기준이 아니다.
                   문턱은 P36 판정 가능 전체의 5번째 백분위수(본 분석과 같은 규칙)이고 35개월 법인에는 P36 값을 그대로 쓴다.
  C  둘 다
탐지·층·RR·군집 부트스트랩은 detect_cnt / validate_cnt / robust_cnt 를 그대로 재사용한다.
"""
from __future__ import annotations

import warnings

import numpy as np

import detect_cnt as dc
import robust_cnt as rb
import validate_cnt as vc

EPS = 1e-9


# --------------------------------------------------------------------------- 대칭 결과 변수
def alt_outcome(Y: np.ndarray, e_range, post=(1, 6), base=(-5, 0), paired: bool | None = None, need_post: int | None = None, need_base: int | None = None) -> np.ndarray:
    """G(i,e) = (post 구간 평균 - base 구간 평균) / base 구간 평균.  구간은 기준월 e 에 대한 상대 월 (post=(1,6) 은 e+1..e+6).

    base 와 post 의 길이가 같으면(기본) 본 분석의 결과 g 와 같은 **쌍 맞춤(M0)**: 두 시점 모두 관측된 달끼리만 평균하고 짝이 길이-1 미만이면 NaN.
    길이가 다르면(예: base=(-11, 0) 12개월) 각 구간에서 관측된 달만 평균하며 각 구간 길이-1 개월 이상 관측돼야 한다.
    base 평균이 0 이하이면 정의하지 않는다(NaN). e_range 밖의 열은 NaN."""
    n, T_ = Y.shape
    G = np.full((n, T_), np.nan)
    lp, lb = post[1] - post[0] + 1, base[1] - base[0] + 1
    paired = (lp == lb) if paired is None else paired
    if paired and lp != lb:
        raise ValueError("쌍 맞춤은 두 구간의 길이가 같아야 합니다.")
    need_post = lp - 1 if need_post is None else need_post
    need_base = lb - 1 if need_base is None else need_base
    for e in e_range:
        if paired:
            c, o, _ = vc._paired(Y, e, post[0], post[1], post[0] - base[0], lp - 1)
        else:
            a, b, ba, bb = e + post[0], e + post[1], e + base[0], e + base[1]
            if ba < 0 or a < 0 or b >= T_:
                continue
            cur, old = Y[:, a : b + 1], Y[:, ba : bb + 1]
            ok = ((~np.isnan(cur)).sum(axis=1) >= need_post) & ((~np.isnan(old)).sum(axis=1) >= need_base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = np.where(ok, np.nanmean(cur, axis=1), np.nan)
                o = np.where(ok, np.nanmean(old, axis=1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            G[:, e] = np.where(o > 0, (c - o) / o, np.nan)
    return G


def units_with_outcome(D: dict, G: np.ndarray, theta: float):
    """준비된 표본 D(vc.prepare 결과)의 탐지·층을 그대로 두고 결과만 G(<= theta 이면 약화)로 바꾼 단위와 층. 반환 (units, (cell_u, cell_grp))."""
    U = vc.build_units(D["det"], D["status"], G, theta, D["size"], D["sector"], D["er"])
    return U, vc.cells_main(U)


def run_scenario(D: dict, F: int, B: int, seed: int, base: dict | None = None, G: np.ndarray | None = None, theta: float | None = None) -> dict:
    """한 시나리오의 RR·95% 범위(robust_cnt.run_variant 와 같은 형식). G·theta 를 주면 결과 정의만 바꾼다. base 를 주면 같은 재표집으로 짝지은 '기준 대비 비'."""
    if G is None:
        return rb.run_variant(D, F, B, seed, base)
    U, cells = units_with_outcome(D, G, theta)
    return rb.run_variant(D, F, B, seed, base, cells=cells, units=U)


# --------------------------------------------------------------------------- 탐지 비교 (후보월 확장)
def first_detect_by_month(det: np.ndarray, months) -> dict:
    """기준월별 처음 신호가 잡힌 기업 수 {e: 수} (months 에 든 달만)."""
    det = np.asarray(det)
    cnt = np.bincount(det[det >= 0], minlength=dc.T)
    return {int(e): int(cnt[e]) for e in months}


def compare_detection(det_base: np.ndarray, det_ext: np.ndarray) -> dict:
    """확장 전후의 최초 신호: 신호 기업 수, 새로 잡힌 기업(확장에서만), 더 이른 달로 앞당겨진 기업, 그대로인 기업, 사라진 기업(있으면 오류)."""
    b, x = np.asarray(det_base), np.asarray(det_ext)
    both = (b >= 0) & (x >= 0)
    return {"n_base": int((b >= 0).sum()), "n_ext": int((x >= 0).sum()), "new": int(((b < 0) & (x >= 0)).sum()),
            "earlier": int((both & (x < b)).sum()), "same": int((both & (x == b)).sum()), "lost": int(((b >= 0) & (x < 0)).sum()),
            "later": int((both & (x > b)).sum())}


def pk(res: dict) -> dict:
    """run_variant 결과를 저장용 요약으로 (nb07 과 같은 형식)."""
    R = res["R"]
    d = {k: (None if not np.isfinite(R[k]) else float(R[k])) for k in ("rr", "lo", "hi", "p_det", "p_ctrl")}
    d["n_det"] = int(res["n_det"])
    d["ratio"] = None if "ratio" not in res else [float(x) for x in res["ratio"]]
    d["jaccard"] = None if not res.get("overlap") else float(res["overlap"]["jaccard"])
    return d
