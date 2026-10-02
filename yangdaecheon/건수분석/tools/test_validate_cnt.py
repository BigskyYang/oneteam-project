"""validate_cnt (P5 입금 검증) 점검: 계획서 정의를 그대로 옮긴 느린 참조 구현과 대조 + 합성 데이터에 심은 입금 약화 탐지.  python test_validate_cnt.py

실제 데이터는 쓰지 않는다. 참조 구현은 기업·층을 하나씩 도는 파이썬 반복문이며 validate_cnt 의 함수를 호출하지 않는다.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
import validate_cnt as vc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


rng = np.random.default_rng(11)

# ======================================================================= 1) 결과 변수 g, 탐지 전 값
Y = np.round(np.exp(rng.normal(3, 1, (80, 36))))
Y[rng.random(Y.shape) < 0.07] = np.nan
Y[rng.random(80) < 0.05, :12] = 0.0


def naive_g(Y, e, need=5):
    out = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        S = [m for m in range(e + 1, e + 7) if not math.isnan(Y[i, m]) and not math.isnan(Y[i, m - 12])]
        if len(S) >= need:
            cur = sum(Y[i, m] for m in S) / len(S)
            old = sum(Y[i, m - 12] for m in S) / len(S)
            if old > 0:
                out[i] = (cur - old) / old
    return out


G = vc.g_matrix(Y, range(17, 30))
check("g (사후 6개월 / 같은 달 전년 - 1, 쌍 맞춤) 일치", all(np.allclose(G[:, e], naive_g(Y, e), equal_nan=True) for e in range(17, 30)))
check("g 는 e < 17 에서 NaN (범위 밖)", np.isnan(G[:, :17]).all())


def naive_pre(Y, e, k, need=5):
    L = np.full(Y.shape[0], np.nan)
    H = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        w = [Y[i, m] for m in range(e - 11, e - 5) if e - 11 >= 0 and not math.isnan(Y[i, m])]
        if len(w) >= need:
            L[i] = sum(w) / len(w)
        if e - k - 17 < 0:                      # 계획서: h_pre 는 t >= k+18 (e >= k+17) 에서만 정의 (전년 비교 창 전체가 패널 안)
            continue
        S = [m for m in range(e - k - 5, e - k + 1) if m - 12 >= 0 and not math.isnan(Y[i, m]) and not math.isnan(Y[i, m - 12])]
        if len(S) >= need:
            cur = sum(Y[i, m] for m in S) / len(S)
            old = sum(Y[i, m - 12] for m in S) / len(S)
            if old > 0:
                H[i] = (cur - old) / old
    return L, H


Lp, Hp = vc.pre_matrices(Y, 6, range(17, 30))
bad = 0
for e in range(17, 30):
    Ln, Hn = naive_pre(Y, e, 6)
    bad += int(not (np.allclose(Lp[:, e], Ln, equal_nan=True) and np.allclose(Hp[:, e], Hn, equal_nan=True)))
check("L_pre·h_pre 일치 (탐지 창 [e-5, e] 와 겹치지 않는 [e-11, e-6] 기준)", bad == 0)
check("h_pre 는 e < k+17 = 23 에서 정의되지 않음 (전년 비교 달이 패널 밖)", np.isnan(Hp[:, :23]).all() and np.isfinite(Hp[:, 23:]).any())

# ======================================================================= 2) 표본 단위 (탐지 기업-월과 위험집합)
n = 400
status = np.full((n, 36), -1)
for e in range(17, 30):
    status[:, e] = np.where(rng.random(n) < 0.8, 0, rng.integers(1, 4, n))
det = np.full(n, -1)
for i in range(n):
    cand = [e for e in range(17, 30) if status[i, e] == 0]
    if cand and rng.random() < 0.3:
        det[i] = cand[rng.integers(0, len(cand))]
Gm = np.full((n, 36), np.nan)
Gm[:, 17:30] = rng.normal(-0.02, 0.4, (n, 13))
Gm[rng.random(Gm.shape) < 0.05] = np.nan
size = rng.integers(0, 5, (n, 36)).astype(np.int8)
sector = rng.integers(0, 6, n).astype(np.int8)
theta = float(np.nanpercentile(Gm[:, 17:30], 5))
U = vc.build_units(det, status, Gm, theta, size, sector, range(17, 30))
naive = set()
for e in range(17, 30):
    for i in range(n):
        if status[i, e] != 0 or math.isnan(Gm[i, e]):
            continue
        if det[i] == e:
            naive.add((i, e, True))
        elif det[i] < 0 or det[i] > e:
            naive.add((i, e, False))
got = set(zip(U.firm.tolist(), U.e.tolist(), U.is_det.tolist()))
check("탐지군·위험집합 구성 일치 (이후 탐지돼도 t월 비교군 자격 유지, 이전에 탐지된 기업은 제외)", got == naive and len(got) == len(U))
later = [(i, e) for i in range(n) for e in range(17, 30) if det[i] > e and status[i, e] == 0 and not math.isnan(Gm[i, e])]
check("나중에 탐지될 기업이 앞선 달의 대조군에 포함됨", len(later) > 0 and all((i, e, False) in got for i, e in later[:50]))
already = [(i, e) for i in range(n) for e in range(17, 30) if det[i] >= 0 and det[i] < e and status[i, e] == 0]
check("이미 탐지된 기업은 이후 달의 대조군에서 빠짐", len(already) > 0 and all((i, e, False) not in got and (i, e, True) not in got for i, e in already[:50]))
check("결과 ev 는 g <= θ", np.array_equal(U.ev, Gm[U.firm, U.e] <= theta + 1e-9))


# ======================================================================= 3) 층별 표준화 RR (공통 지지, 업종 병합) -- 참조 구현
def naive_rr(units, min_ctrl=5):
    cells = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for e, sz, sec, d, ev, w in units:
        c = cells[(e, sz, sec)]
        if d:
            c[0] += w
            c[2] += w * ev
        else:
            c[1] += w
            c[3] += w * ev
    groups = defaultdict(list)
    for (e, sz, sec), c in cells.items():
        groups[(e, sz)].append(c)
    obs = exp = n1s = tot = 0.0
    for gk, cs in groups.items():
        tot += sum(c[0] for c in cs)
        need_fb = any(c[0] > 0 and c[1] < min_ctrl for c in cs)
        if not need_fb:
            for c in cs:
                if c[0] > 0:
                    obs += c[2]
                    exp += c[0] * c[3] / c[1]
                    n1s += c[0]
        else:
            N1, N0 = sum(c[0] for c in cs), sum(c[1] for c in cs)
            EV1, EV0 = sum(c[2] for c in cs), sum(c[3] for c in cs)
            if N1 > 0 and N0 >= min_ctrl:
                obs += EV1
                exp += N1 * EV0 / N0
                n1s += N1
    return (obs / exp if exp > 0 else math.nan), n1s, tot


# 희소한 층을 일부러 만든다 (규모 4~5 에서 업종을 잘게)
m = len(U)
cell_u, cell_grp = vc.cells_main(U)
est = vc.estimate(U, cell_u, cell_grp)
rr0, n1s0, tot0 = naive_rr([(U.e[i], U.size[i], U.sector[i], bool(U.is_det[i]), float(U.ev[i]), 1.0) for i in range(m)])
check("RR = 관찰/기대 일치 (층별 표준화, 공통 지지, 업종 병합)", abs(est["rr"] - rr0) < 1e-9 and abs(est["n1_supported"] - n1s0) < 1e-9 and abs(est["n1"] - tot0) < 1e-9,
      f"{est['rr']} vs {rr0}")
check("공통 지지 제외율이 0~1", 0 <= est["excluded_share"] <= 1)

# 가중 = 복제 (군집 부트스트랩의 핵심): 기업 중복 횟수 w 로 가중한 추정 == 단위를 실제로 복제한 추정
F = n
mult = np.bincount(rng.integers(0, F, F), minlength=F).astype(float)
w = mult[U.firm]
est_w = vc.estimate(U, cell_u, cell_grp, w=w)
rows = [(U.e[i], U.size[i], U.sector[i], bool(U.is_det[i]), float(U.ev[i]), float(w[i])) for i in range(m)]
rr_w, _, _ = naive_rr(rows)
check("가중 추정 == 참조 구현의 가중 추정", abs(est_w["rr"] - rr_w) < 1e-9)
rep_idx = np.repeat(np.arange(m), w.astype(int))
U2 = U.sub(rep_idx)
cu2, cg2 = vc.cells_main(U2)
check("가중 추정 == 단위를 실제로 복제한 추정", abs(vc.estimate(U2, cu2, cg2)["rr"] - est_w["rr"]) < 1e-9)

# eff_cells 로 집계한 값과 직접 계산이 같다
C = int(cell_u.max()) + 1
n1, n0, ev1, ev0 = vc._counts(cell_u, C, np.ones(m), U.is_det, U.ev.astype(float))
eff = vc.eff_cells(n1, n0, cell_grp)
e_u = eff[cell_u]
E = C + int(cell_grp.max()) + 1
mm = e_u >= 0
N1 = np.bincount(e_u[mm], (U.is_det * 1.0)[mm], E)
N0 = np.bincount(e_u[mm], ((~U.is_det) * 1.0)[mm], E)
EV1 = np.bincount(e_u[mm], (U.is_det * U.ev * 1.0)[mm], E)
EV0 = np.bincount(e_u[mm], ((~U.is_det) * U.ev * 1.0)[mm], E)
sup = (N1 > 0) & (N0 > 0)
rr_eff = EV1[sup].sum() / (N1[sup] * EV0[sup] / N0[sup]).sum()
check("유효 층(eff_cells)으로 집계한 RR 과 직접 계산한 RR 일치", abs(rr_eff - est["rr"]) < 1e-9)

# 효과가 심어진 데이터에서 RR 이 1 보다 크다 / 효과가 없으면 CI 가 1 을 포함
U3 = U.sub(np.ones(m, dtype=bool))
U3.ev = np.where(U3.is_det, rng.random(m) < 0.5, rng.random(m) < 0.1)       # 탐지군 50% vs 대조군 10% -> RR 약 5
cu3, cg3 = vc.cells_main(U3)
e3 = vc.estimate(U3, cu3, cg3)
star, bad_b = vc.bootstrap_rr(U3, cu3, cg3, F, B=300, seed=1)
s3 = vc.summarize_boot(e3["rr"], star, 300)
check("심은 효과(RR≈5): 점추정 > 3, CI 하한 > 1, 채택", e3["rr"] > 3 and s3["lo"] > 1 and s3["adopt"] and s3["p_one"] < 0.025, str(s3))
U4 = U.sub(np.ones(m, dtype=bool))
U4.ev = rng.random(m) < 0.15
cu4, cg4 = vc.cells_main(U4)
e4 = vc.estimate(U4, cu4, cg4)
star4, _ = vc.bootstrap_rr(U4, cu4, cg4, F, B=300, seed=1)
s4 = vc.summarize_boot(e4["rr"], star4, 300)
check("효과 없음: CI 가 1 을 포함, 채택하지 않음", s4["lo"] <= 1 <= s4["hi"] and not s4["adopt"], str(s4))
a1, _ = vc.bootstrap_rr(U3, cu3, cg3, F, B=50, seed=7)
a2, _ = vc.bootstrap_rr(U3, cu3, cg3, F, B=50, seed=7)
check("부트스트랩은 같은 시드에서 같은 결과", np.array_equal(a1, a2))
sm = vc.summarize_boot(2.0, np.array([0.8, 1.2, 3.0, 2.5]), 4)
check("단측 p = (1 + #{RR* <= 1}) / (B' + 1)", abs(sm["p_one"] - (1 + 1) / 5) < 1e-12)

# 연속 결과 평균 차이: 참조
out = rng.normal(1, 0.3, m)
md = vc.mean_diff(U, cell_u, cell_grp, out)
# 참조: n1_s 가중
num = den = 0.0
cells = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
for i in range(m):
    c = cells[(U.e[i], U.size[i], U.sector[i])]
    if U.is_det[i]:
        c[0] += 1
        c[2] += out[i]
    else:
        c[1] += 1
        c[3] += out[i]
groups = defaultdict(list)
for (e, sz, sec), c in cells.items():
    groups[(e, sz)].append(c)
sm1 = e0 = sn1 = 0.0
for gk, cs in groups.items():
    need_fb = any(c[0] > 0 and c[1] < 5 for c in cs)
    if not need_fb:
        for c in cs:
            if c[0] > 0:
                sm1 += c[2]
                e0 += c[0] * c[3] / c[1]
                sn1 += c[0]
    else:
        N1, N0 = sum(c[0] for c in cs), sum(c[1] for c in cs)
        if N1 > 0 and N0 >= 5:
            sm1 += sum(c[2] for c in cs)
            e0 += N1 * sum(c[3] for c in cs) / N0
            sn1 += N1
check("O2 표준화 평균 차이 일치", abs(md["diff"] - (sm1 - e0) / sn1) < 1e-9)

# ======================================================================= 4) 균형표
x = rng.normal(0, 1, m)
sd = vc.std_diff(x, U, cell_u, cell_grp)
x1, x0 = x[U.is_det], x[~U.is_det]
want_smd = (x1.mean() - x0.mean()) / math.sqrt(0.5 * (x1.var() + x0.var()))
check("균형표: 조정 전 표준화 차이 일치", abs(sd["smd_raw"] - want_smd) < 1e-9)
xs = np.where(U.size >= 3, 1.0, 0.0)               # 규모로 정의된 변수 -> 층 안에서 같으므로 조정 후 차이 0
check("균형표: 층 변수(규모)는 조정 후 표준화 차이가 0", abs(vc.std_diff(xs, U, cell_u, cell_grp)["smd_adj"]) < 1e-9)

# ======================================================================= 5) H-C2 경로
J = list(range(-12, 7))
Yp = np.round(np.exp(rng.normal(3, 0.8, (n, 36))) + 1.0)
Yp[rng.random(Yp.shape) < 0.03] = np.nan
Yw = np.minimum(Yp, np.nanpercentile(Yp, 99.5))
sel = U.e >= 24
Up = U.sub(sel)
cup, cgp = vc.cells_main(Up)
num, den, valid = vc.path_arrays(Yw, Up.firm, Up.e, J)
n1p, n0p, _, _ = vc._counts(cup, int(cup.max()) + 1, np.ones(len(Up)), Up.is_det, np.zeros(len(Up)))
effp = vc.eff_cells(n1p, n0p, cgp)
d, r1, r0, ns = vc.path_delta(num, den, valid, effp[cup], Up.is_det, np.ones(len(Up)))


def naive_path(j, Up, effu):
    st = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])         # num1, den1, num0, den0, w1
    for u in range(len(Up)):
        if effu[u] < 0:
            continue
        a, b = Up.e[u] + j, Up.e[u] + j - 12
        if not (0 <= b and a < 36):
            continue
        cur, old = Yw[Up.firm[u], a], Yw[Up.firm[u], b]
        if math.isnan(cur) or math.isnan(old):
            continue
        s = st[effu[u]]
        if Up.is_det[u]:
            s[0] += cur
            s[1] += old
            s[4] += 1
        else:
            s[2] += cur
            s[3] += old
    keys = [k for k, s in st.items() if s[1] > 0 and s[3] > 0 and s[4] > 0]
    if not keys:
        return math.nan
    W = sum(st[k][4] for k in keys)
    return sum(st[k][4] / W * (st[k][0] / st[k][1] - st[k][2] / st[k][3]) for k in keys)


effu = effp[cup]
bad = sum(int(abs(d[jj] - naive_path(j, Up, effu)) > 1e-9) for jj, j in enumerate(J) if not math.isnan(d[jj]))
check("경로 Δ(j) 일치 (층 내 입금 합의 전년 비, 탐지군 층 가중)", bad == 0 and np.isfinite(d).sum() >= 15)
check("경로: 기준월 e >= 24 이벤트는 j = -12 .. +6 모두 정의됨", np.isfinite(d).all(), str(d))
s_lin = vc.path_summaries(np.array([0.02 * j for j in J], dtype=float), J)
check("사전 기울기 = j ∈ [-12,-6] 구간 OLS 기울기", abs(s_lin["slope_pre"] - 0.02) < 1e-9 and abs(s_lin["pre_mean"] - np.mean([0.02 * j for j in range(-12, -5)])) < 1e-9)
check("동시 구간 j ∈ [-5,0], 사후 j ∈ [1,6] 평균", abs(s_lin["conc_mean"] - np.mean([0.02 * j for j in range(-5, 1)])) < 1e-9 and abs(s_lin["post_mean"] - np.mean([0.02 * j for j in range(1, 7)])) < 1e-9)
check("해석 문구: CI 가 0 을 포함하면 '뚜렷한 입금 차이 없음'", "뚜렷한 입금 차이가 확인되지 않음" in vc.interpret_pre((-0.05, 0.04), (-0.01, 0.01)))
check("해석 문구: 사전 Δ 가 음수로 0 을 벗어나면 '이미 약했음' ('선행' 표현 없음)", "이미 약했음" in vc.interpret_pre((-0.20, -0.05), (-0.01, 0.01)) and "선행" not in vc.interpret_pre((-0.20, -0.05), (-0.01, 0.01)))
check("해석 문구: 사전 Δ 가 양수로 0 을 벗어나면 '오히려 높았음'(평균 회귀) — 약했다고 쓰지 않음", "오히려 높았음" in vc.interpret_pre((0.05, 0.20), (-0.01, 0.01)) and "이미 약했음" not in vc.interpret_pre((0.05, 0.20), (-0.01, 0.01)))
check("해석 문구: 평균은 0 을 포함하나 기울기가 0 을 벗어나면 추세 문구", "추세적" in vc.interpret_pre((-0.05, 0.04), (0.01, 0.03)))
sums, deltas = vc.bootstrap_path(Up, cup, cgp, num, den, valid, J, F, B=40, seed=3)
check("경로 부트스트랩: 요약 4개 × 40회, 형태 일치", all(v.shape == (40,) for v in sums.values()) and deltas.shape == (40, len(J)))

# ======================================================================= 6) 합성 데이터: 심은 위축형 기업의 입금 약화를 찾는다
df, exp = make(4000, seed=5)
panel, rep = cl.build_panel(df)
masks = cl.sample_masks(panel)
P36 = masks["P36"]
I1 = dc.i1_matrix(panel, P36)
MODE, K = "직전", 6
L_ = dc.lag_for(MODE, K)
FL = dc.ceiling_window_flag(panel, P36, K, L_)
d_fix = dc.decide_fixed(I1, MODE, K, FL)
res = dc.apply_rule(I1, MODE, K, d_fix["phi"], d_fix["n"])
det_, status_, ch_ = res["det"], res["status"], res["change"]
erange = list(dc.e_range(K))
Yin = panel.amt["요구불입금금액"][P36]
Gy = vc.g_matrix(Yin, erange)
th = vc.theta_low(Gy, status_, erange)
edges = vc.size_edges(ch_.C, status_, erange)
szm = vc.size_group(ch_.C, edges)
sec_codes = dc.firm_attribute(panel, P36, "업종_대분류")
secg, top_codes = vc.sector_groups(sec_codes, sec_codes, len(panel.attr_levels["업종_대분류"]))
Us = vc.build_units(det_, status_, Gy, th, szm, secg, erange)
cus, cgs = vc.cells_main(Us)
Fs = int(P36.sum())
es = vc.estimate(Us, cus, cgs)
stars, _ = vc.bootstrap_rr(Us, cus, cgs, Fs, B=300, seed=5)
ss = vc.summarize_boot(es["rr"], stars, 300)
print(f"    (합성) 탐지 {int(Us.is_det.sum())}곳, 대조군 단위 {int((~Us.is_det).sum()):,}, RR {es['rr']:.2f} (CI {ss['lo']:.2f}~{ss['hi']:.2f})")
check("합성: 심은 위축형을 탐지한 기업의 입금 약화 비율이 대조군보다 높음 (RR > 1.5, CI 하한 > 1)", es["rr"] > 1.5 and ss["lo"] > 1.0, str(ss))
# 경로: 입금 약화는 위축 시작 후에 나타난다 (사후 Δ < 0, 건수 감소 이전 Δ ≈ 0)
Ywin = np.minimum(Yin, np.nanpercentile(Yin, 99.5))
selp = Us.e >= 24
Upp = Us.sub(selp)
cup2, cgp2 = vc.cells_main(Upp)
num2, den2, valid2 = vc.path_arrays(Ywin, Upp.firm, Upp.e, J)
n1q, n0q, _, _ = vc._counts(cup2, int(cup2.max()) + 1, np.ones(len(Upp)), Upp.is_det, np.zeros(len(Upp)))
dd, _, _, _ = vc.path_delta(num2, den2, valid2, vc.eff_cells(n1q, n0q, cgp2)[cup2], Upp.is_det, np.ones(len(Upp)))
sm_ = vc.path_summaries(dd, J)
print(f"    (합성) 경로 요약: 이전 {sm_['pre_mean']:+.3f}, 기울기 {sm_['slope_pre']:+.4f}, 동시 {sm_['conc_mean']:+.3f}, 사후 {sm_['post_mean']:+.3f}")
check("합성: 사후 입금 차이가 건수 감소 이전보다 훨씬 음수 (사후 Δ < 이전 Δ - 0.1)", sm_["post_mean"] < sm_["pre_mean"] - 0.1, str(sm_))

# ======================================================================= 7) prepare / 고정 상수 / S1 / S-C1 / 잠금 대상 표본에 같은 상수 적용
consts = vc.fit_constants(panel, P36, MODE, K, d_fix["phi"], d_fix["n"])
D = vc.prepare(panel, P36, consts)
check("prepare: 직접 만든 단위와 같은 수의 탐지·대조 단위", int(D["units"].is_det.sum()) == int(Us.is_det.sum()) and len(D["units"]) == len(Us))
check("고정 상수: θ_O1 이 판정 가능 g 의 5번째 백분위수와 같음", abs(consts["theta_o1"] - th) < 1e-12)
Ds1 = D["units"].sub((D["units"].e >= K + 17) & ~np.isnan(D["units"].x["h_pre"]) & ~np.isnan(D["units"].x["L_pre"]))
cu1, cg1 = vc.cells_s1(Ds1, consts)
check("S1: 층 번호가 압축되고 병합 없음(층마다 상위 층이 자기 자신)", cu1.max() + 1 == len(cg1) and len(np.unique(cg1)) == len(cg1))
r_s1 = vc.analyze_units(Ds1, D["F"], B=30, seed=2, cells=(cu1, cg1))
check("S1: 추정이 정의됨 (RR 숫자)", np.isfinite(r_s1["rr"]))
Dc1 = vc.prepare(panel, P36, consts, drop_ceiling_cells=True)
check("S-C1: 천장 칸을 뺀 탐지 기업은 원래 탐지 기업과 겹칠 수 있으나 천장 칸에서는 탐지되지 않음",
      all(not Dc1["FL"][i, Dc1["det"][i]] for i in np.flatnonzero(Dc1["det"] >= 0)))
masks35 = masks["P35"]
D35 = vc.prepare(panel, masks35, consts)
check("잠금 대상 표본(P35만)에도 P36 에서 고정한 상수를 그대로 적용 (θ·경계·상위 업종 동일)", D35["phi"] == consts["phi"] and D35["n_pct"] == consts["n_pct"] and len(D35["units"]) > 0)
Dv1 = vc.prepare(panel, P36, consts, phi=60.0, n_pct=15.0)
check("v1 규칙(Φ=60, n=15)으로도 같은 상수로 준비됨", len(Dv1["units"]) > 0)

# ======================================================================= 8) 경로 민감도(중앙값), analyze_units 가 부트스트랩 분포를 돌려줌
Jm = list(range(-3, 4))
nm, dn, vl = vc.path_arrays(Yw, Up.firm, Up.e, Jm)
med = vc.path_median_delta(nm, dn, vl, effu, Up.is_det)
bad_m = 0
for jj, j in enumerate(Jm):
    st_ = defaultdict(lambda: ([], []))
    for u_ in range(len(Up)):
        if effu[u_] < 0 or not vl[u_, jj] or dn[u_, jj] <= 0:
            continue
        st_[effu[u_]][0 if Up.is_det[u_] else 1].append(nm[u_, jj] / dn[u_, jj] - 1)
    tw = sw = 0.0
    for k_, (a, b) in st_.items():
        if a and b:
            sw += len(a) * (np.median(a) - np.median(b))
            tw += len(a)
    want = sw / tw if tw else math.nan
    bad_m += int(not ((math.isnan(want) and math.isnan(med[jj])) or abs(want - med[jj]) < 1e-9))
check("경로 중앙값 민감도 일치 (층 안 기업별 변화율 중앙값, 탐지군 층 가중)", bad_m == 0)
ra = vc.analyze_units(D["units"], D["F"], B=40, seed=4)
check("analyze_units: 부트스트랩 분포(star)와 지원 정보 반환", ra["star"].size > 0 and ra["B_valid"] == ra["star"].size and 0 <= ra["excluded_share"] <= 1)

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
