"""robust_cnt (P7 민감도) 점검: 계획서 정의를 그대로 옮긴 느린 참조 구현과 대조.  python test_robust_cnt.py

실제 데이터는 쓰지 않는다(합성 데이터). 참조 구현은 반복문이며 robust_cnt·validate_cnt 의 같은 함수를 호출하지 않는다.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
import robust_cnt as rb  # noqa: E402
import validate_cnt as vc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


df, exp = make(3000, seed=9)
panel, rep = cl.build_panel(df)
masks = cl.sample_masks(panel)
P36 = masks["P36"]
MODE, K = "직전", 6
I1 = dc.i1_matrix(panel, P36)
FL = dc.ceiling_window_flag(panel, P36, K, dc.lag_for(MODE, K))
d_fix = dc.decide_fixed(I1, MODE, K, FL)
consts = vc.fit_constants(panel, P36, MODE, K, d_fix["phi"], d_fix["n"])
D0 = vc.prepare(panel, P36, consts)
F = D0["F"]


def same_units(a, b):
    return (len(a) == len(b) and (a.firm == b.firm).all() and (a.e == b.e).all() and (a.is_det == b.is_det).all() and (a.ev == b.ev).all())


# ======================================================================= 1) 기본값 불변, 지수 변형
check("prepare 의 새 인자(cap·I·post·theta_q)를 기본값으로 주면 결과가 이전과 같음", same_units(D0["units"], vc.prepare(panel, P36, consts, cap=cl.DEFAULT_CAP, post=(1, 6))["units"]))
check("I 를 직접 넘겨도(I1 그대로) 결과가 같음", same_units(D0["units"], vc.prepare(panel, P36, consts, I=I1)["units"]))
I51, I100 = dc.i1_matrix(panel, P36, 51.0), dc.i1_matrix(panel, P36, 100.0)
check("cap 51 ≤ cap 60 ≤ cap 100 (같은 칸 기준, 미관측은 NaN)", np.nanmax(I51 - I1) <= 1e-9 and np.nanmin(I100 - I1) >= -1e-9 and np.isnan(I51).sum() == np.isnan(I1).sum())
lo, hi = rb.i1_bound(panel, P36, "low"), rb.i1_bound(panel, P36, "high")
bad = 0
rng = np.random.default_rng(1)
obs = panel.observed[P36]
for _ in range(400):
    i, t = rng.integers(0, obs.shape[0]), rng.integers(0, 36)
    if not obs[i, t]:
        continue
    cds = [int(panel.cnt[c][P36][i, t]) for c in cl.CH5]
    if min(cds) < 0:
        continue
    wl = sum(cl.BAND_LOWER[c] for c in cds)
    wh = sum((cl.BAND_UPPER + [100.0])[c] for c in cds)
    bad += int(abs(lo[i, t] - wl) > 1e-9 or abs(hi[i, t] - wh) > 1e-9)
check("구간 하한·상한 지수(I1_low, I1_high=50건초과 100 시나리오) 참조 구현과 일치", bad == 0)
check("I1_low ≤ I1 ≤ I1_high", np.nanmax(lo - I1) <= 1e-9 and np.nanmin(hi - I1) >= -1e-9)
I2 = rb.i2_matrix(panel, P36)
bad = 0
for _ in range(300):
    i, t = rng.integers(0, obs.shape[0]), rng.integers(0, 36)
    cds = [int(panel.cnt[c][P36][i, t]) for c in cl.CH5]
    want = sum(cds) if (obs[i, t] and min(cds) >= 0) else math.nan
    bad += int(not ((math.isnan(want) and math.isnan(I2[i, t])) or abs(want - I2[i, t]) < 1e-9))
check("I2(순서 코드 합) 참조 구현과 일치", bad == 0)
# I2 로 문턱을 다시 구해 돌린다 (B 변형)
d2 = dc.decide_fixed(I2, MODE, K, FL)
C2 = vc.fit_constants(panel, P36, MODE, K, d2["phi"], d2["n"], I=I2)
D2 = vc.prepare(panel, P36, C2, phi=d2["phi"], n_pct=d2["n"], I=I2)
check("I2 변형: 문턱을 지표에 맞춰 다시 구한 규칙으로 탐지 기업이 존재", int((D2["det"] >= 0).sum()) > 0 and d2["n"] >= 5, str(d2["n"]))

# ======================================================================= 2) 입금 약화 정의 변형
Y = panel.amt["요구불입금금액"][P36]
ER = D0["er"]


def naive_g3(Y, e, need=2):
    out = np.full(Y.shape[0], np.nan)
    for i in range(Y.shape[0]):
        S = [m for m in range(e + 1, e + 4) if m < 36 and not math.isnan(Y[i, m]) and not math.isnan(Y[i, m - 12])]
        if len(S) >= need:
            cur = sum(Y[i, m] for m in S) / len(S)
            old = sum(Y[i, m - 12] for m in S) / len(S)
            if old > 0:
                out[i] = (cur - old) / old
    return out


G3 = vc.g_matrix(Y, ER, 2, (1, 3))
check("추적 3개월 g(최소 2쌍)가 참조 구현과 일치", all(np.allclose(G3[:, e], naive_g3(Y, e), equal_nan=True) for e in ER))
D3 = vc.prepare(panel, P36, consts, post=(1, 3))
want_theta = float(np.percentile(np.concatenate([G3[:, e][(D3["status"][:, e] == 0) & ~np.isnan(G3[:, e])] for e in ER]), 5))
check("추적 3개월 변형: 입금 약화 문턱을 P36 판정 가능 전체에서 같은 방식(5번째 백분위수)으로 다시 구함", abs(D3["theta"] - want_theta) < 1e-12 and abs(D3["theta"] - consts["theta_o1"]) > 1e-9)
D10 = vc.prepare(panel, P36, consts, theta_q=10.0)
want10 = float(np.percentile(np.concatenate([D0["G"][:, e][(D0["status"][:, e] == 0) & ~np.isnan(D0["G"][:, e])] for e in ER]), 10))
D25 = vc.prepare(panel, P36, consts, theta_q=2.5)
check("하위 10%·2.5% 변형: 문턱이 해당 백분위수이고 10% 문턱이 5% 문턱보다 높음(약화 정의가 넓어짐)",
      abs(D10["theta"] - want10) < 1e-12 and D10["theta"] > D0["theta"] > D25["theta"])
check("정의를 넓히면 약화 사건이 늘고 좁히면 줄어듦 (탐지 단위는 같음)", int(D10["units"].ev.sum()) > int(D0["units"].ev.sum()) > int(D25["units"].ev.sum()) and int(D10["units"].is_det.sum()) == int(D0["units"].is_det.sum()))

# ======================================================================= 3) 탐지 일치도, 층 대안, 보조 비교군
det_a = np.array([-1, 17, 18, 20, -1, 25])
det_b = np.array([-1, 17, 19, -1, 22, 25])
ov = rb.detection_overlap(det_a, det_b)
check("탐지 일치도: Jaccard = 교집합/합집합, 같은 기업 탐지월 차이 중앙값·같은 달 비율",
      abs(ov["jaccard"] - 3 / 5) < 1e-12 and ov["n_a"] == 4 and ov["n_b"] == 4 and ov["n_both"] == 3 and abs(ov["month_diff_median"]) < 1e-12 and abs(ov["same_month_share"] - 2 / 3) < 1e-12, str(ov))


def naive_rr_keyed(u, key_cell, key_grp):
    cells = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    grp_of = {}
    for i in range(len(u)):
        k = key_cell(i)
        grp_of[k] = key_grp(i)
        c = cells[k]
        if u.is_det[i]:
            c[0] += 1
            c[2] += u.ev[i]
        else:
            c[1] += 1
            c[3] += u.ev[i]
    groups = defaultdict(list)
    for k, g in grp_of.items():
        groups[g].append(k)
    obs = exp_ = 0.0
    for g, keys in groups.items():
        fb = any(cells[k][0] > 0 and cells[k][1] < vc.MIN_CTRL for k in keys)
        if not fb:
            for k in keys:
                n1, n0, e1, e0 = cells[k]
                if n1 > 0:
                    obs += e1
                    exp_ += n1 * e0 / n0 if n0 > 0 else 0.0
        else:
            n1 = sum(cells[k][0] for k in keys)
            n0 = sum(cells[k][1] for k in keys)
            if n1 > 0 and n0 >= vc.MIN_CTRL:
                obs += sum(cells[k][2] for k in keys)
                exp_ += n1 * sum(cells[k][3] for k in keys) / n0
    return obs / exp_ if exp_ > 0 else math.nan


U = D0["units"]
cu1, cg1 = rb.cells_pooled_months(U)
want1 = naive_rr_keyed(U, lambda i: (U.size[i], U.sector[i]), lambda i: U.size[i])
got1 = vc.estimate(U, cu1, cg1)["rr"]
check("층 대안 1(기준월 풀링 x 규모 x 업종군, 병합 = 규모) RR 이 참조 구현과 일치", abs(want1 - got1) < 1e-9, f"{want1} vs {got1}")
cu2, cg2 = rb.cells_no_sector(U)
want2 = naive_rr_keyed(U, lambda i: (U.e[i], U.size[i]), lambda i: (U.e[i], U.size[i]))
got2 = vc.estimate(U, cu2, cg2)["rr"]
check("층 대안 2(기준월 x 규모, 업종군 없음) RR 이 참조 구현과 일치", abs(want2 - got2) < 1e-9, f"{want2} vs {got2}")
Un = rb.never_detected_controls(U, D0["det"])
ctrl_firms = Un.firm[~Un.is_det]
check("보조 비교군: 탐지 단위는 그대로, 대조 단위는 한 번도 탐지되지 않은 기업의 것만",
      int(Un.is_det.sum()) == int(U.is_det.sum()) and (D0["det"][ctrl_firms] < 0).all() and len(Un) < len(U))
check("보조 비교군은 위험집합의 부분집합(이후 탐지될 기업의 대조 단위가 빠짐)", int((~Un.is_det).sum()) < int((~U.is_det).sum()))

# ======================================================================= 4) 변형 실행·쌍 부트스트랩 비
base = rb.run_variant(D0, F, B=60, seed=5)
base_obj = {"R": base["R"], "D": D0}
same = rb.run_variant(D0, F, B=60, seed=5, base=base_obj)
check("run_variant: 기준 자신과 비교하면 변형 RR ÷ 기준 RR = 1 (같은 기업 재표집으로 짝지음)", "ratio" in same and abs(same["ratio"][0] - 1) < 1e-12 and abs(same["ratio"][1] - 1) < 1e-9 and abs(same["ratio"][2] - 1) < 1e-9, str(same.get("ratio")))
v51 = rb.run_variant(vc.prepare(panel, P36, consts, cap=51.0), F, B=60, seed=5, base=base_obj)
check("run_variant(cap 51): RR 과 비, 탐지 일치도(Jaccard)가 계산됨", np.isfinite(v51["R"]["rr"]) and "ratio" in v51 and 0 < v51["overlap"]["jaccard"] <= 1)
check("비 = 변형 RR ÷ 기준 RR 의 점추정", abs(v51["ratio"][0] - v51["R"]["rr"] / base["R"]["rr"]) < 1e-12)

# ======================================================================= 5) ② 비교 기준·③ 창 길이 선택을 최종 n 으로 다시
I_ = dc.i1_matrix(panel, P36)
sea = dc.seasonality(I_)
check("choose_mode: thr 를 안 주면 기존 임시 기준(30%)과 같음", dc.choose_mode(I_, sea)["mode"] == dc.choose_mode(I_, sea, thr=0.30)["mode"] and dc.choose_mode(I_, sea)["rates"] == dc.choose_mode(I_, sea, thr=0.30)["rates"])
check("transient_share: thr 를 안 주면 기존과 같음", dc.transient_share(I_, 3, "직전") == dc.transient_share(I_, 3, "직전", thr=0.30))
check("transient_share: thr 를 올리면 탐지 기업이 줄어듦", dc.transient_share(I_, 3, "직전", thr=0.35)["detected"] <= dc.transient_share(I_, 3, "직전", thr=0.30)["detected"])
s30 = rb.redo_selection(panel, P36, 0.30)
dd = dc.decide(I_)
check("redo_selection(30%) 가 기존 전체 절차(decide)의 비교 기준·k 와 같음", s30["mode"] == dd["mode"] and s30["k"] == dd["k"], f"{s30['mode']}{s30['k']} vs {dd['mode']}{dd['k']}")
s35 = rb.redo_selection(panel, P36, 0.35)
check("redo_selection(35%): 비교 기준, k(후보 중 하나), Φ>0, n>=5 가 정해짐", s35["mode"] in dc.MODES and s35["k"] in dc.KS and s35["phi"] > 0 and s35["n"] >= 5, str({k: v for k, v in s35.items() if k != "shares"}))

# ======================================================================= 6) 표본 변형이 같은 코드로 돌아감
Dk3 = None
d3 = dc.decide_fixed(I1, MODE, 3, dc.ceiling_window_flag(panel, P36, 3, dc.lag_for(MODE, 3)))
C3 = vc.fit_constants(panel, P36, MODE, 3, d3["phi"], d3["n"])
Dk3 = vc.prepare(panel, P36, C3)
rk3 = rb.run_variant(Dk3, F, B=40, seed=5)
check("k=3 변형: 이 k 의 Φ·n 으로 다시 구해 RR 계산", np.isfinite(rk3["R"]["rr"]) and len(Dk3["er"]) > len(ER))
Dp = vc.prepare(panel, masks["P35p"], consts)
rp = rb.run_variant(Dp, Dp["F"], B=40, seed=6)
check("P35+ (M0) 변형: 같은 규칙·상수로 RR 계산", np.isfinite(rp["R"]["rr"]))
U24, U25 = U.sub(U.e <= 23), U.sub(U.e >= 24)
check("기간 분할: 2024 기준월과 2025 기준월의 탐지 기업이 겹치지 않고 합이 전체와 같음",
      int(U24.is_det.sum()) + int(U25.is_det.sum()) == int(U.is_det.sum()) and not (set(U24.firm[U24.is_det].tolist()) & set(U25.firm[U25.is_det].tolist())))
r24 = vc.analyze_units(U24, F, B=40, seed=7)
check("기간 분할: 기간별 RR 계산", np.isfinite(r24["rr"]))
print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
