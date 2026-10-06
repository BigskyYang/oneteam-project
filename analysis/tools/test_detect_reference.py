"""detect_cnt 의 벡터화 구현을 **계획서 수식을 그대로 옮긴 느린 참조 구현**과 대조한다 (같은 입력, 빈 달·동률 포함).  python test_detect_reference.py

참조 구현은 기업·월을 하나씩 도는 파이썬 반복문이며 detect_cnt 의 함수를 호출하지 않는다 (cntlib.fill_causal 만 M1 에서 공유).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


rng = np.random.default_rng(5)
n = 300
base = np.exp(rng.normal(2.5, 1.2, (n, 1))) * np.exp(rng.normal(0, 0.35, (n, 36)))
I = np.round(base)                                   # 정수 -> 동률 많음
I[rng.random(I.shape) < 0.05] = np.nan               # 빈 달
I[rng.random(n) < 0.05, 6:18] = 0.0                  # 일부 기업은 비교 창이 모두 0 (판정 불가 (a))
I[:5, 20:] = I[:5, 20:] * 0.3                        # 위축 흉내


# ---- 참조 구현 (계획서 4b 그대로)
def ref_change(I, k, L, need):
    n_, T_ = I.shape
    r = np.full((n_, T_), np.nan)
    C = np.full((n_, T_), np.nan)
    cnt = np.zeros((n_, T_), int)
    for i in range(n_):
        for e in range(k - 1 + L, T_):
            S = [m for m in range(e - k + 1, e + 1) if not math.isnan(I[i, m]) and not math.isnan(I[i, m - L])]
            cnt[i, e] = len(S)
            if len(S) >= need:
                R_ = sum(I[i, m] for m in S) / len(S)
                C_ = sum(I[i, m - L] for m in S) / len(S)
                C[i, e] = C_
                if C_ > 0:
                    r[i, e] = (R_ - C_) / C_
    return r, C, cnt


def ref_status(r, C, cnt, need, phi, k):
    n_, T_ = r.shape
    st = np.full((n_, T_), -1, dtype=int)
    for i in range(n_):
        for e in range(k + 11, 30):
            if cnt[i, e] < need:
                st[i, e] = 3
            elif C[i, e] <= 0:
                st[i, e] = 1
            elif C[i, e] < phi:
                st[i, e] = 2
            else:
                st[i, e] = 0
    return st


def ref_first_detect(r, C, phi, thr, k):
    n_, _ = r.shape
    out = np.full(n_, -1)
    for i in range(n_):
        for e in range(k + 11, 30):
            if not math.isnan(r[i, e]) and C[i, e] >= phi and r[i, e] <= -thr + 1e-9:
                out[i] = e
                break
    return out


for mode in dc.MODES:
    for k in dc.KS:
        L = dc.lag_for(mode, k)
        need = cl.min_pairs(k)
        ch = dc.change_matrix(I, k, L)
        r0, C0, c0 = ref_change(I, k, L, need)
        check(f"변화율·비교 평균·짝 수 일치 ({mode}, k={k})",
              np.allclose(ch.r, r0, equal_nan=True) and np.allclose(ch.C, C0, equal_nan=True) and np.array_equal(ch.cnt, c0))
        for phi in (0.0, 8.0):
            st = dc.status_matrix(ch, phi)
            check(f"판정 상태 일치 ({mode}, k={k}, Φ={phi})", np.array_equal(st, ref_status(r0, C0, c0, need, phi, k)))
            for thr in (0.25, 0.5):
                check(f"최초 탐지월 일치 ({mode}, k={k}, Φ={phi}, n={thr * 100:.0f}%)",
                      np.array_equal(dc.first_detect(ch, phi, thr), ref_first_detect(r0, C0, phi, thr, k)))

# 사후 변화율: 참조 = 창 [e+1, e+k] 를 작년 같은 달과 직접 비교
k = 3
rp = dc.post_change(I, k)
need = cl.min_pairs(k)
bad = 0
for i in rng.integers(0, n, 60):
    for e in range(k + 11, 30):
        S = [m for m in range(e + 1, e + k + 1) if not math.isnan(I[i, m]) and not math.isnan(I[i, m - 12])]
        want = math.nan
        if len(S) >= need:
            R_ = np.mean([I[i, m] for m in S])
            C_ = np.mean([I[i, m - 12] for m in S])
            want = (R_ - C_) / C_ if C_ > 0 else math.nan
        got = rp[i, e]
        bad += int(not ((math.isnan(want) and math.isnan(got)) or abs(want - got) < 1e-9))
check("사후 변화율(작년 같은 달 대비) 일치", bad == 0, str(bad))


# ---- 활동량 하한 Φ: 독립 구현 (정렬·반복문으로 분위 경계, 병합, IQR)
def ref_phi(C, r, min_bin=100, n_bins=10):
    pairs = [(c, x) for c, x in zip(C.ravel(), r.ravel()) if not math.isnan(x) and c > 0]
    pairs.sort()
    cs = [p[0] for p in pairs]
    m = len(cs)

    def quant(q):                                       # 선형 보간 분위수를 정렬 배열에서 직접 계산
        pos = q * (m - 1)
        lo_, hi_ = int(math.floor(pos)), int(math.ceil(pos))
        return cs[lo_] + (cs[hi_] - cs[lo_]) * (pos - lo_)

    edges = sorted(set(quant(j / n_bins) for j in range(1, n_bins)))
    groups = [[] for _ in range(len(edges) + 1)]
    for c, x in pairs:
        b = sum(1 for ed in edges if c >= ed)           # 경계값과 같으면 위쪽 분위
        groups[b].append((c, x))
    groups = [g for g in groups if g]
    changed = True
    while changed and len(groups) > 1:                  # 분위당 min_bin 미만이면 위쪽과 병합(마지막이면 아래쪽)
        changed = False
        for j, g in enumerate(groups):
            if len(g) < min_bin:
                t = j + 1 if j + 1 < len(groups) else j - 1
                groups[t] = sorted(groups[t] + g)
                del groups[j]
                changed = True
                break
    B = len(groups)
    iqr = [float(np.subtract(*np.percentile([x for _, x in g], [75, 25]))) for g in groups]
    top = math.ceil(B / 2)
    ref = float(np.median(iqr[B - top:]))
    for d in range(B):
        if all(v <= 1.5 * ref + 1e-9 for v in iqr[d:]):
            return (0.0 if d == 0 else float(min(c for c, _ in groups[d]))), B
    return None, B


for mode, k in (("전년", 3), ("직전", 6), ("직전", 2)):
    ch = dc.change_matrix(I, k, dc.lag_for(mode, k))
    lo = k + 11
    Cw, rw = ch.C[:, lo:30], ch.r[:, lo:30]
    for use_excl in (False, True):
        r_use = rw
        if use_excl:
            ex = rng.random(rw.shape) < 0.3
            r_use = np.where(ex, np.nan, rw)
        phi_ref, B_ref = ref_phi(Cw, r_use)
        phi_dc, info = dc.decile_floor(Cw, r_use)
        check(f"활동량 하한 Φ 일치 ({mode}, k={k}, 일부 칸 제외={use_excl})",
              phi_ref is not None and abs(phi_ref - phi_dc) < 1e-9 and B_ref == info["bins"], f"ref={phi_ref} dc={phi_dc} bins {B_ref}/{info['bins']}")

# ---- n: 참조 = 정렬 후 직접 선형 보간한 5번째 백분위수
r_all = rng.normal(-0.02, 0.2, 5000)
s_ = np.sort(r_all)
pos = 0.05 * (len(s_) - 1)
q = s_[int(math.floor(pos))] + (s_[int(math.ceil(pos))] - s_[int(math.floor(pos))]) * (pos - math.floor(pos))
n0 = -100 * q
want_n = max(5.0, math.floor(n0 / 5 + 0.5) * 5.0)
check("n 일치 (정렬 후 직접 선형 보간)", dc.n_from_r(r_all)["n"] == want_n and abs(dc.n_from_r(r_all)["n0"] - n0) < 1e-9)

# ---- 일시적 하락 비율: 참조 = 기업별 반복문
mode, k = "직전", 3
L = dc.lag_for(mode, k)
need = cl.min_pairs(k)
r0, C0, c0 = ref_change(I, k, L, need)
det = ref_first_detect(r0, C0, 0.0, dc.PROV_THR, k)
tr = dfn = 0
for i in np.flatnonzero(det >= 0):
    e = det[i]
    S = [m for m in range(e + 1, e + k + 1) if not math.isnan(I[i, m]) and not math.isnan(I[i, m - 12])]
    if len(S) >= need:
        R_ = np.mean([I[i, m] for m in S])
        C_ = np.mean([I[i, m - 12] for m in S])
        if C_ > 0:
            dfn += 1
            tr += int((R_ - C_) / C_ >= dc.TRANS_THR)
res = dc.transient_share(I, k, mode)
check("일시적 하락 비율의 분자·분모 일치", res["defined"] == dfn and res["transient"] == tr and res["detected"] == int((det >= 0).sum()), f"{res} vs {tr}/{dfn}")

# ---- M1: 참조 = 후보월마다 fill_causal 로 채운 뒤 기업별 계산 (월 > s 는 NaN 이라 쓰이지 않음)
codes = rng.integers(0, 10, (120, 36)).astype(float)
codes[rng.random(codes.shape) < 0.04] = np.nan
cnt_codes = np.where(np.isnan(codes), -1, codes).astype(np.int8)
panel = cl.CountPanel(months=np.arange(36), observed=~np.isnan(codes), cnt={c: cnt_codes.copy() for c in cl.CNT_COLS}, amt={}, attr={},
                      attr_levels={}, band_labels=[], synthetic=True)
k, L = 3, 12
need = cl.min_pairs(k)
phi, thr = 20.0, 0.30
det_dc = dc.first_detect_m1(panel, np.ones(120, dtype=bool), k, L, phi, thr)
Z = cl.codes_float(cnt_codes, panel.observed)
det_ref = np.full(120, -1)
for e in range(k + 11, 30):
    Zs = cl.fill_causal(Z, e)
    A = cl.approx_count(Zs, cl.DEFAULT_CAP) * len(cl.CH5)        # 5채널이 같은 값이므로 I1 = 5 x 한 채널
    for i in range(120):
        if det_ref[i] >= 0 or math.isnan(Zs[i, e]):
            continue
        S = [m for m in range(e - k + 1, e + 1) if not math.isnan(A[i, m]) and not math.isnan(A[i, m - L])]
        if len(S) < need:
            continue
        R_ = np.mean([A[i, m] for m in S])
        C_ = np.mean([A[i, m - L] for m in S])
        if C_ > 0 and C_ >= phi and (R_ - C_) / C_ <= -thr + 1e-9:
            det_ref[i] = e
check("M1 최초 탐지월 일치 (후보월별 인과적 채움, 기업별 반복문)", np.array_equal(det_dc, det_ref), f"{(det_dc != det_ref).sum()}개 다름")

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
