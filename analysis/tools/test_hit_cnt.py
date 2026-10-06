"""hit_cnt (P9 보강 검증) 점검: 정의를 그대로 옮긴 느린 참조 구현과 대조.  python test_hit_cnt.py

실제 데이터는 쓰지 않는다(합성 데이터). 참조 구현은 반복문이며 hit_cnt 의 같은 함수를 호출하지 않는다.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
import hit_cnt as ht  # noqa: E402
import validate_cnt as vc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


# ======================================================================= 1) 2x2 표 (손으로 만든 작은 예 + 무작위 대조)
class FakeUnits:
    def __init__(self, is_det, ev):
        self.is_det, self.ev = np.asarray(is_det, bool), np.asarray(ev, bool)


u = FakeUnits([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
t = ht.two_by_two(u)
check("손 계산 예: a=2 b=2 c=1 d=5", (t["a"], t["b"], t["c"], t["d"]) == (2, 2, 1, 5))
check("손 계산 예: 적중 2/4, 비신호 약화율 1/6, 재현율 2/3, 놓침 1/3, lift (1/2)/(1/6)=3",
      abs(t["precision"] - 0.5) < 1e-12 and abs(t["ctrl_rate"] - 1 / 6) < 1e-12 and abs(t["recall"] - 2 / 3) < 1e-12 and abs(t["miss_share"] - 1 / 3) < 1e-12 and abs(t["lift_raw"] - 3.0) < 1e-12)
rng = np.random.default_rng(3)
bad = 0
for _ in range(200):
    n = int(rng.integers(5, 60))
    dt, ev = rng.random(n) < 0.3, rng.random(n) < 0.25
    t = ht.two_by_two(FakeUnits(dt, ev))
    a = sum(1 for i in range(n) if dt[i] and ev[i])
    b = sum(1 for i in range(n) if dt[i] and not ev[i])
    c = sum(1 for i in range(n) if (not dt[i]) and ev[i])
    d = sum(1 for i in range(n) if (not dt[i]) and not ev[i])
    ok = (t["a"], t["b"], t["c"], t["d"]) == (a, b, c, d) and t["a"] + t["b"] + t["c"] + t["d"] == n
    if a + b and c + d:
        ok &= abs(t["precision"] - a / (a + b)) < 1e-12 and abs(t["ctrl_rate"] - c / (c + d)) < 1e-12
    if a + c:
        ok &= abs(t["recall"] - a / (a + c)) < 1e-12 and abs(t["miss_share"] - (1 - a / (a + c))) < 1e-12
    bad += int(not ok)
check("무작위 200회: 칸 수·비율이 반복문과 일치, 칸 합 = 전체", bad == 0)
t0 = ht.two_by_two(FakeUnits([0, 0, 0], [1, 0, 0]))
check("신호가 없으면 적중 비율은 NaN (0 으로 나누지 않음)", math.isnan(t0["precision"]) and t0["recall"] == 0.0)
m = ht.monthly_averages({"n_signal": 26, "a": 6, "c": 20}, 13)
check("월평균: 신호 2, 약화 2, 잡은 수 6/13", abs(m["signals_per_month"] - 2) < 1e-12 and abs(m["weak_per_month"] - 2) < 1e-12 and abs(m["caught_per_month"] - 6 / 13) < 1e-12)

# ======================================================================= 2) 기업 단위 신호 시점 (손으로 만든 상태 행렬 + 반복문 참조)
ER = list(range(17, 30))


def fake_D(n, seed):
    r = np.random.default_rng(seed)
    st = np.full((n, 36), -1, dtype=np.int8)
    st[:, 17:30] = r.choice([0, 0, 0, 2], size=(n, 13))
    G = r.normal(0, 1, size=(n, 36))
    G[r.random((n, 36)) < 0.05] = np.nan
    det = np.where(r.random(n) < 0.4, r.integers(17, 30, n), -1)
    return {"er": ER, "status": st, "G": G, "theta": -1.2, "det": det}


def ref_timing(D):
    th = D["theta"]
    out = {"n_judgeable": 0, "n_weak": 0, "n_signal": 0, "n_signal_weak": 0, "early": 0, "late": 0, "none": 0}
    for i in range(D["G"].shape[0]):
        months = [e for e in ER if D["status"][i, e] == 0]
        out["n_judgeable"] += int(bool(months))
        weak_m = [e for e in months if not math.isnan(D["G"][i, e]) and D["G"][i, e] <= th + 1e-9]
        sig = D["det"][i] >= 0
        out["n_signal"] += int(sig)
        if weak_m:
            out["n_weak"] += 1
            out["n_signal_weak"] += int(sig)
            first = min(weak_m)
            if sig and D["det"][i] <= first:
                out["early"] += 1
            elif sig:
                out["late"] += 1
            else:
                out["none"] += 1
    return out


bad = 0
for sd in range(8):
    D = fake_D(400, sd)
    a, b = ht.firm_timing(D), ref_timing(D)
    bad += int(a != b or a["early"] + a["late"] + a["none"] != a["n_weak"])
check("기업 단위 신호 시점: 8개 무작위 행렬에서 반복문과 일치, 먼저+늦음+없음 = 약화 기업", bad == 0)
D = {"er": ER, "status": np.zeros((2, 36), np.int8), "G": np.full((2, 36), 0.5), "theta": -1.0, "det": np.array([20, -1])}
D["G"][0, 22] = -2.0            # 기업 0: 첫 약화 22, 신호 20 -> 먼저
D["G"][1, 18] = -2.0            # 기업 1: 약화 18, 신호 없음 -> 신호 없음
f = ht.firm_timing(D)
check("손 계산 예: 신호가 약화보다 먼저 1곳, 신호 없음 1곳", f["early"] == 1 and f["none"] == 1 and f["late"] == 0 and f["n_weak"] == 2)
D["det"] = np.array([25, -1])
check("손 계산 예: 신호(25)가 첫 약화(22)보다 늦으면 '늦음'", ht.firm_timing(D)["late"] == 1)

# ======================================================================= 3) 신호 비교 달과 약화 기준 달의 겹침
ro6 = ht.reference_overlap(6, 6)
ro12 = ht.reference_overlap(6, 12)
check("직전 6개월 비교(L=6): 신호의 비교 달이 약화 기준 달과 6개월 모두 겹침", ro6["overlap_months"] == 6 and ro6["signal_reference"] == ro6["outcome_base"])
check("전년 같은 달 비교(L=12): 겹치는 달 없음 (비교 달 [e-17, e-12] vs 기준 달 [e-11, e-6])", ro12["overlap_months"] == 0 and ro12["signal_reference"] == [3, 8] and ro12["outcome_base"] == [9, 14])
check("모든 후보월에서 같은 결론 (e=17..29)", all(ht.reference_overlap(6, 6, e)["overlap_months"] == 6 and ht.reference_overlap(6, 12, e)["overlap_months"] == 0 for e in range(17, 30)))

# ======================================================================= 4) 합성 패널에서 전년 규칙·적중 표
df, exp = make(3000, seed=9)
panel, rep = cl.build_panel(df)
masks = cl.sample_masks(panel)
P36 = masks["P36"]
MODE, K = "직전", 6
I1 = dc.i1_matrix(panel, P36)
FL = dc.ceiling_window_flag(panel, P36, K, dc.lag_for(MODE, K))
d_fix = dc.decide_fixed(I1, MODE, K, FL)
C_main = vc.fit_constants(panel, P36, MODE, K, d_fix["phi"], d_fix["n"])
D_main = vc.prepare(panel, P36, C_main)
d_y, C_y = ht.year_on_year_rule(panel, P36, K)
check("전년 규칙: 비교 기준=전년, k=6, Φ>0, n>=5 가 이 규칙에서 다시 정해짐", d_y["mode"] == "전년" and d_y["k"] == K and d_y["L"] == 12 and d_y["phi"] > 0 and d_y["n"] >= 5 and C_y["mode"] == "전년", str({k: d_y[k] for k in ("mode", "k", "L", "phi", "n")}))
D_y = vc.prepare(panel, P36, C_y)
check("전년 규칙으로 준비한 표본은 L=12 변화율을 쓴 탐지임", D_y["change"].L == 12 and D_main["change"].L == K)
# 참조 구현: 전년 규칙의 최초 탐지월 (반복문, 같은 Φ·n 사용)
need = cl.min_pairs(K)
bad, checked = 0, 0
obs_I = I1
for i in range(0, I1.shape[0], 7):
    want = -1
    for e in range(K + 11, dc.E_MAX + 1):
        cur = [(obs_I[i, m], obs_I[i, m - 12]) for m in range(e - K + 1, e + 1) if not math.isnan(obs_I[i, m]) and not math.isnan(obs_I[i, m - 12])]
        if len(cur) < need:
            continue
        R, Cc = float(np.mean([x for x, _ in cur])), float(np.mean([y for _, y in cur]))
        if Cc > 0 and Cc >= d_y["phi"] and (R - Cc) / Cc <= -d_y["n"] / 100.0 + 1e-9:
            want = e
            break
    bad += int(want != D_y["det"][i])
    checked += 1
check(f"전년 규칙의 최초 탐지월이 반복문 참조 구현과 일치 ({checked}개 기업)", bad == 0)
# 겹침이 없다는 점이 실제 행렬에서도 성립: 전년 신호의 비교 평균은 약화 기준 달(e-11..e-6)의 입금과 무관한 달만 쓴다 (달 번호로 확인)
check("전년 규칙의 비교 달 [e-17, e-12] 는 약화 기준 달 [e-11, e-6] 과 겹치지 않음 (모든 후보월)", all(max(range(e - 17, e - 11)) < min(range(e - 11, e - 5)) for e in range(17, 30)))

# 적중 표
tt = ht.two_by_two(D_main["units"])
check("합성: 2x2 칸 합이 단위 수와 같고 적중 비율 > 비신호 약화율 (심어 둔 입금 약화)", tt["a"] + tt["b"] + tt["c"] + tt["d"] == len(D_main["units"]) and tt["precision"] > tt["ctrl_rate"], str(tt))
ft = ht.firm_timing(D_main)
check("합성: 기업 단위 요약의 신호 기업 수 = 탐지 기업 수, 약화 기업 >= 신호&약화 기업", ft["n_signal"] == int((D_main["det"] >= 0).sum()) and ft["n_weak"] >= ft["n_signal_weak"], str(ft))
ns = sorted({15.0, float(d_fix["n"]), 70.0})
grid = ht.precision_recall_grid(panel, P36, C_main, ns)
sig = [g["n_signal"] for g in grid]
im = ns.index(float(d_fix["n"]))
check("기준 n 격자: 감소 기준이 클수록 신호 기업-월이 줄거나 같음", all(sig[i] >= sig[i + 1] for i in range(len(sig) - 1)), str(sig))
check("기준 n 격자: 합성 P36 의 결정 n 행이 직접 계산한 2x2 와 같음", (grid[im]["a"], grid[im]["b"], grid[im]["c"], grid[im]["d"]) == (tt["a"], tt["b"], tt["c"], tt["d"]), f"{ns} {grid[im]}")
ry = vc.analyze_units(D_y["units"], D_y["F"], B=40, seed=11)
check("전년 규칙으로도 RR 이 계산됨(합성에서 연관 유지: RR > 1)", np.isfinite(ry["rr"]) and ry["rr"] > 1.0, f"rr={ry['rr']}")
print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
