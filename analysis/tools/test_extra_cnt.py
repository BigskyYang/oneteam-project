"""extra_cnt (P10 추가 민감도: 후보월 확장·대칭 결과 변수) 점검: 정의를 그대로 옮긴 느린 참조 구현과 대조.  python test_extra_cnt.py

실제 데이터는 쓰지 않는다(합성 데이터). 참조 구현은 반복문이며 extra_cnt·detect_cnt 의 같은 함수를 호출하지 않는다.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
import extra_cnt as ex  # noqa: E402
import robust_cnt as rb  # noqa: E402
import validate_cnt as vc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


# ======================================================================= 1) 후보월 범위 (e_lo)
check("기본 후보월: k=6 이면 e=17..29 (2024.06~2025.06, 13개월), k=3 이면 14..29", list(dc.e_range(6)) == list(range(17, 30)) and list(dc.e_range(3)) == list(range(14, 30)))
check("e_lo 를 주면 그 달부터: e_lo=11 이면 11..29 (2023.12~2025.06, 19개월)", list(dc.e_range(6, 11)) == list(range(11, 30)) and len(dc.e_range(6, 11)) == 19)
check("e_lo_for: 기본 k+11, 직전(L=6) 은 11 까지 허용, 전년(L=12) 은 k+11 이 한계", dc.e_lo_for(6, 6) == 17 and dc.e_lo_for(6, 6, 11) == 11 and dc.e_lo_for(6, 12) == 17)
for k_, L_, lo_ in ((6, 6, 10), (6, 12, 16), (3, 3, 4)):
    try:
        dc.e_lo_for(k_, L_, lo_)
        raised = False
    except cl.CntError:
        raised = True
    check(f"e_lo_for: 변화율이 정의되지 않는 이른 달(k={k_}, L={L_}, e_lo={lo_}) 은 오류", raised)

rng = np.random.default_rng(5)
n, T = 400, dc.T
base = rng.uniform(5, 60, size=(n, 1))
I = np.maximum(base * (1 + rng.normal(0, 0.25, size=(n, T))), 0.0)
drop_at = rng.integers(8, 28, size=n)
for i in range(n):
    if rng.random() < 0.35:
        I[i, drop_at[i] :] *= rng.uniform(0.2, 0.6)               # 일부 기업은 중간에 거래가 줄어든다
I[rng.random((n, T)) < 0.03] = np.nan                              # 빈 달
K, MODE = 6, "직전"
PHI, NPCT = 10.0, 35.0
ch = dc.change_matrix(I, K, dc.lag_for(MODE, K))
det_def = dc.first_detect(ch, PHI, NPCT / 100)
check("e_lo=None 과 e_lo=17(기본값을 직접 줌) 의 최초 탐지가 같음", np.array_equal(det_def, dc.first_detect(ch, PHI, NPCT / 100, 17)))
det_ext = dc.first_detect(ch, PHI, NPCT / 100, 11)
check("후보월을 넓히면 기존에 탐지된 기업은 모두 탐지되고 탐지월은 같거나 더 이름", ((det_def < 0) | ((det_ext >= 0) & (det_ext <= det_def))).all())
check("확장에서만 새로 탐지되는 기업이 있음 (무작위 자료에서 탐지 기업 수가 늘어남)", int((det_ext >= 0).sum()) > int((det_def >= 0).sum()))
check("확장에서 탐지된 기업은 모두 e>=11", ((det_ext < 0) | (det_ext >= 11)).all())
cmp_ = ex.compare_detection(det_def, det_ext)
check("compare_detection: 사라지거나 늦춰진 기업이 없고 합이 맞음", cmp_["lost"] == 0 and cmp_["later"] == 0 and cmp_["new"] + cmp_["earlier"] + cmp_["same"] == cmp_["n_ext"] and cmp_["n_base"] == int((det_def >= 0).sum()))

# 참조 구현(반복문): 최초 탐지월
need = cl.min_pairs(K)


def ref_first(i, lo):
    for e in range(lo, dc.E_MAX + 1):
        pairs = [(I[i, m], I[i, m - K]) for m in range(e - K + 1, e + 1) if not math.isnan(I[i, m]) and not math.isnan(I[i, m - K])]
        if len(pairs) < need:
            continue
        R = float(np.mean([a for a, _ in pairs]))
        C = float(np.mean([b for _, b in pairs]))
        if C > 0 and C >= PHI and (R - C) / C <= -NPCT / 100.0 + 1e-9:
            return e
    return -1


bad = sum(1 for i in range(n) if ref_first(i, 11) != det_ext[i] or ref_first(i, 17) != det_def[i])
check(f"최초 탐지월(기본·확장)이 반복문 참조 구현과 일치 ({n}개 기업)", bad == 0)
st_def, st_ext = dc.status_matrix(ch, PHI), dc.status_matrix(ch, PHI, 11)
check("판정 상태: 기본은 e<17 이 모두 -1(판정 기간 밖), 확장은 e=11..16 이 판정 기간 안(0~3) 이고 e=10 은 밖", (st_def[:, :17] == -1).all() and (st_ext[:, 11:17] >= 0).all() and (st_ext[:, :11] == -1).all())
check("판정 상태: e>=17 은 기본과 확장이 같음", np.array_equal(st_def[:, 17:], st_ext[:, 17:]))
res_x = dc.apply_rule(I, MODE, K, PHI, NPCT, 11)
check("apply_rule(e_lo) 가 first_detect·status_matrix 와 같은 값", np.array_equal(res_x["det"], det_ext) and np.array_equal(res_x["status"], st_ext))
fm = ex.first_detect_by_month(det_ext, range(11, 30))
check("first_detect_by_month: 월별 합 = 탐지 기업 수, 모든 키가 11..29", sum(fm.values()) == int((det_ext >= 0).sum()) and sorted(fm) == list(range(11, 30)))

# ======================================================================= 2) 대칭 결과 변수 (손으로 만든 예 + 반복문 참조)
Y1 = np.full((1, 36), 100.0)
Y1[0, 6:] = 50.0
g = ex.alt_outcome(Y1, [5], post=(1, 6), base=(-5, 0))
check("손 계산(직전 6개월 대비): e=5, 직전 [0..5]=100, 직후 [6..11]=50 → -0.5", abs(g[0, 5] + 0.5) < 1e-12)
g12 = ex.alt_outcome(Y1, [11], post=(1, 6), base=(-11, 0))
check("손 계산(직전 12개월 대비): e=11, 직전 [0..11] 평균 75, 직후 [12..17]=50 → -1/3", abs(g12[0, 11] + 1 / 3) < 1e-12)
check("정의된 열은 e_range 안뿐 (그 밖은 NaN)", np.isnan(g[0, [0, 4, 6, 11]]).all())
Y2 = Y1.copy()
Y2[0, 0:6] = np.nan
check("직전 구간이 모두 빈 달이면 NaN", np.isnan(ex.alt_outcome(Y2, [5])[0, 5]))
Y3 = Y1.copy()
Y3[0, 3] = np.nan
check("쌍 맞춤: 빈 달 1개는 허용(짝 5개), 짝지은 달만 평균", abs(ex.alt_outcome(Y3, [5])[0, 5] + 0.5) < 1e-12)
Y4 = Y1.copy()
Y4[0, 2:4] = np.nan
check("쌍 맞춤: 빈 달 2개면 짝이 4개라 NaN", np.isnan(ex.alt_outcome(Y4, [5])[0, 5]))
Y5 = np.zeros((1, 36))
Y5[0, 6:] = 30.0
check("직전 평균이 0 이하이면 NaN (0 으로 나누지 않음)", np.isnan(ex.alt_outcome(Y5, [5])[0, 5]))
check("관측 구간이 자료 밖이면 NaN (e=2 는 직전 [-3..2] 가 시작 전, e=33 은 직후가 끝을 넘음)", np.isnan(ex.alt_outcome(Y1, [2, 33])[0, [2, 33]]).all())

rngy = np.random.default_rng(8)
Yr = np.abs(rngy.normal(100, 40, size=(60, 36)))
Yr[rngy.random(Yr.shape) < 0.06] = np.nan
Yr[rngy.random(Yr.shape) < 0.01] = 0.0


def ref_g(row, e, post, base, paired):
    lp, lb = post[1] - post[0] + 1, base[1] - base[0] + 1
    if paired:
        lag = post[0] - base[0]
        ps = [(row[m], row[m - lag]) for m in range(e + post[0], e + post[1] + 1) if 0 <= m - lag and m < 36 and not math.isnan(row[m]) and not math.isnan(row[m - lag])]
        if e + post[1] >= 36 or e + post[0] - lag < 0 or len(ps) < lp - 1:
            return math.nan
        c, o = float(np.mean([a for a, _ in ps])), float(np.mean([b for _, b in ps]))
    else:
        if e + base[0] < 0 or e + post[1] >= 36:
            return math.nan
        cv = [row[m] for m in range(e + post[0], e + post[1] + 1) if not math.isnan(row[m])]
        ov = [row[m] for m in range(e + base[0], e + base[1] + 1) if not math.isnan(row[m])]
        if len(cv) < lp - 1 or len(ov) < lb - 1:
            return math.nan
        c, o = float(np.mean(cv)), float(np.mean(ov))
    return (c - o) / o if o > 0 else math.nan


for nm, post_, base_, paired_ in (("직전 6개월(쌍 맞춤)", (1, 6), (-5, 0), True), ("직전 12개월", (1, 6), (-11, 0), False), ("직전 3개월(쌍 맞춤)", (1, 3), (-2, 0), True)):
    er_ = list(range(11, 30))
    G = ex.alt_outcome(Yr, er_, post_, base_)
    bad = 0
    for i in range(Yr.shape[0]):
        for e in er_:
            w = ref_g(Yr[i], e, post_, base_, paired_)
            v = G[i, e]
            bad += int(not ((math.isnan(w) and math.isnan(v)) or (not math.isnan(w) and not math.isnan(v) and abs(w - v) < 1e-9)))
    check(f"alt_outcome({nm}) 가 반복문 참조 구현과 일치 (60개 기업 x 19개월)", bad == 0)

# ======================================================================= 3) 합성 패널 통합
df, exp = make(3000, seed=9)
panel, rep = cl.build_panel(df)
masks = cl.sample_masks(panel)
P36 = masks["P36"]
I1 = dc.i1_matrix(panel, P36)
FL = dc.ceiling_window_flag(panel, P36, K, dc.lag_for(MODE, K))
d_fix = dc.decide_fixed(I1, MODE, K, FL)
C5 = vc.fit_constants(panel, P36, MODE, K, d_fix["phi"], d_fix["n"])
D0 = vc.prepare(panel, P36, C5)
Dx = vc.prepare(panel, P36, C5, e_lo=11)
check("prepare(e_lo=None) 은 기본 후보월(13개월), e_lo=11 은 19개월", len(D0["er"]) == 13 and len(Dx["er"]) == 19 and D0["er"][0] == 17 and Dx["er"][0] == 11)
check("합성: 확장하면 탐지 기업이 늘고 기존 탐지 기업은 모두 유지(탐지월은 같거나 이름)", (lambda c: c["lost"] == 0 and c["later"] == 0 and c["n_ext"] >= c["n_base"])(ex.compare_detection(D0["det"], Dx["det"])))
check("합성: 확장 표본의 단위는 e>=11 이고 기본 표본의 단위는 e>=17", int(Dx["units"].e.min()) >= 11 and int(D0["units"].e.min()) >= 17)

Y = D0["Y"]
G6 = ex.alt_outcome(Y, D0["er"], (1, 6), (-5, 0))
th6 = vc.theta_low(G6, D0["status"], D0["er"], 5.0)
jud = (D0["status"][:, D0["er"]] == 0) & ~np.isnan(G6[:, D0["er"]])
share = float((G6[:, D0["er"]][jud] <= th6 + 1e-9).mean())
check("문턱 = 판정 가능 전체의 5번째 백분위수 → 그 이하 비율이 약 5%", 0.045 <= share <= 0.056, f"{share:.4f}")
U6, cells6 = ex.units_with_outcome(D0, G6, th6)
check("대칭 결과로 다시 만든 단위: 탐지 단위 수는 결과가 정의된 탐지 기업 수 이하, 위험집합이 있음", 0 < int(U6.is_det.sum()) <= int((D0["det"] >= 0).sum()) and int((~U6.is_det).sum()) > 0)
r_base = rb.run_variant(D0, D0["F"], 40, 11)
r_same = ex.run_scenario(D0, D0["F"], 40, 11)
check("run_scenario(G=None) 은 robust_cnt.run_variant 와 같은 RR", abs(r_base["R"]["rr"] - r_same["R"]["rr"]) < 1e-12)
r_sym = ex.run_scenario(D0, D0["F"], 40, 11, {"R": r_base["R"], "D": D0}, G6, th6)
check("합성: 대칭 결과(직전 6개월 대비)로도 RR 이 계산되고 연관 유지(RR > 1, 심어 둔 입금 약화)", np.isfinite(r_sym["R"]["rr"]) and r_sym["R"]["rr"] > 1.0, f"rr={r_sym['R']['rr']}")
r_ext = ex.run_scenario(Dx, Dx["F"], 40, 11, {"R": r_base["R"], "D": D0})
check("합성: 확장 후보월로도 RR 이 계산되고 기준과 탐지 일치도(Jaccard)가 저장됨", np.isfinite(r_ext["R"]["rr"]) and r_ext["R"]["rr"] > 1.0 and r_ext.get("overlap") is not None, str(r_ext.get("overlap")))
pk = ex.pk(r_sym)
check("pk: 저장용 요약에 rr·lo·hi·n_det·ratio 가 있음(JSON 으로 저장 가능)", {"rr", "lo", "hi", "n_det", "ratio"} <= set(pk) and all(isinstance(v, (int, float, list, type(None))) for v in pk.values()))

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
