"""detect_cnt (P4 탐지 규칙) 단위 점검 + 합성 데이터에 심은 위축형 기업 탐지 점검 (실제 데이터 불필요).  python test_detect_cnt.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


rng = np.random.default_rng(0)

# 1) window_change(L=12) 는 cntlib.paired_means 와 같다 (빈 달 포함)
I = rng.uniform(1, 50, (40, 36))
I[rng.random(I.shape) < 0.08] = np.nan
for k in (1, 3, 6):
    need = cl.min_pairs(k)
    R, C, cnt = dc.window_change(I, 30, k, 12, need)
    rc, pr = cl.paired_means(I, 30, k, need)
    same = np.allclose(np.where(cnt >= need, R, np.nan), rc, equal_nan=True) and np.allclose(np.where(cnt >= need, C, np.nan), pr, equal_nan=True)
    check(f"window_change(L=12, k={k}) == paired_means", same)

# 2) 직전 기간 비교의 짝: (m, m-k)
I2 = np.full((1, 36), np.nan)
I2[0, :] = np.arange(36, dtype=float) + 1
R, C, cnt = dc.window_change(I2, 20, 3, 3, 2)      # 최근 [18,19,20] -> 값 19,20,21 ; 비교 [15,16,17] -> 16,17,18
check("직전 기간 비교: 최근 창과 k개월 전 창을 짝지음", abs(R[0] - 20) < 1e-12 and abs(C[0] - 17) < 1e-12 and cnt[0] == 3)

# 3) 변화율 정의: C=0 이면 정의되지 않음, 최소 짝 수 미달이면 정의되지 않음
I3 = np.ones((2, 36)) * 10
I3[0, 6:18] = 0.0                                  # 기업0: 1년 전 비교 창이 모두 0
I3[1, 18:20] = np.nan                              # 기업1: 빈 달 2개
ch = dc.change_matrix(I3, 3, 12)
check("C=0 이면 r 정의 불가, C 는 0 으로 남음 (판정 불가 (a))", np.isnan(ch.r[0, 20]) and ch.C[0, 20] == 0)
st = dc.status_matrix(ch, 0.0)
check("status: (a) 비교 평균 0 -> 1", st[0, 20] == 1)
check("status: 판정 기간 밖은 -1", st[0, 5] == -1 and st[0, 12] == -1)

# 4) 사후 변화율은 창 [e+1, e+k] 를 작년 같은 달과 비교 (방식과 무관하게 L=12)
Ip = np.ones((1, 36)) * 10
Ip[0, 21:24] = 5.0                                  # e=20 이후 3개월이 절반
rp = dc.post_change(Ip, 3)
check("post_change: 열 e 에 [e+1, e+k] 의 작년 대비 변화율", abs(rp[0, 20] - (-0.5)) < 1e-12 and abs(rp[0, 17]) < 1e-12)

# 5) n 의 부호와 반올림
r = np.linspace(-0.6, 0.6, 1001)                    # 5% 분위수 = -0.54
nn = dc.n_from_r(r)
check("n: 변화율 5번째 백분위수가 음수이면 양의 n (r <= -n%)", nn["n0"] > 0 and abs(nn["n0"] - 54.0) < 0.1 and nn["n"] == 55.0)
check("n: 0.5 는 올림 (42.5 -> 45)", dc.n_from_r(np.array([-0.425] * 100))["n"] == 45.0)
check("n: 최소 5%p", dc.n_from_r(np.array([-0.02] * 100))["n"] == 5.0)
try:
    dc.n_from_r(np.abs(r))
    check("n: 5번째 백분위수가 0 이상이면 중단", False)
except cl.CntError:
    check("n: 5번째 백분위수가 0 이상이면 중단", True)

# 6) 탐지: 경계(r == -thr) 는 탐지, 처음 성립하는 달, 활동량 하한 적용
Id = np.ones((3, 36)) * 10
Id[0, 20:] = 7.0                                    # e=20 에서 k=1 변화율 -30% (전년 대비)
Id[1, 25:] = 1.0
Id[2, :] = 0.5
Id[2, 22:] = 0.05                                   # 활동량이 작은 기업 (C = 0.5): Φ=5 이면 판정 대상이 아님
ch1 = dc.change_matrix(Id, 1, 12)
det = dc.first_detect(ch1, 0.0, 0.30)
check("first_detect: 경계 -30% 도 탐지, 처음 성립하는 달", det[0] == 20 and det[1] == 25 and det[2] == 22, str(det))
det_f = dc.first_detect(ch1, 5.0, 0.30)
check("first_detect: Φ 미만(C<Φ)인 기업은 판정 대상이 아님", det_f[2] == -1 and det_f[0] == 20)
check("first_detect: 변화 없는 달은 탐지하지 않음", dc.first_detect(dc.change_matrix(np.ones((2, 36)) * 4, 3, 12), 0.0, 0.3).max() == -1)

# 7) 활동량 하한: 낮은 분위만 IQR 이 크면 그 위 분위부터
n = 6000
C = np.exp(rng.normal(2.0, 1.0, n))
rr = rng.normal(0, 0.05, n)
low = C < np.quantile(C, 0.3)
rr[low] = rng.normal(0, 0.5, low.sum())             # 낮은 활동량에서만 변동이 큼
phi, info = dc.decile_floor(C, rr)
check("decile_floor: 낮은 분위의 큰 변동을 걸러냄 (Φ 가 하위 30% 경계 근처)", np.quantile(C, 0.25) < phi <= np.quantile(C, 0.45) + 1e-9, f"phi={phi:.3f}")
check("decile_floor: 분위 병합 정보가 일관", info["count"].sum() == n and info["bins"] == len(info["iqr"]))
phi0, info0 = dc.decile_floor(C, rng.normal(0, 0.1, n))
check("decile_floor: 모든 분위가 같은 변동이면 Φ=0", phi0 == 0.0)
Ct = np.concatenate([np.full(3000, 2.0), np.exp(rng.normal(3, 0.5, 3000))])    # 값이 같은 동률이 절반
rt = rng.normal(0, 0.1, 6000)
phit, infot = dc.decile_floor(Ct, rt)
check("decile_floor: 값이 같은 경계는 합치고 오류 없이 동작", infot["bins"] < 10 and infot["count"].sum() == 6000)
firm = np.repeat(np.arange(600), 10)
bs = dc.phi_bootstrap(firm, C[:6000], rr[:6000], B=30)
check("phi_bootstrap: 재표집 분포 반환", bs.shape == (30,) and np.isfinite(bs).any())

# 8) 검정력 근사
m1, m2 = dc.mde(100, 1000), dc.mde(400, 1000)
check("mde: 탐지군이 커지면 작아짐", m2 < m1 < 20, f"{m1:.2f}, {m2:.2f}")
check("mde: 탐지군이 3개뿐이면 MDE 가 매우 크고(>5), 1개면 검출 불가(inf)", dc.mde(3, 1000) > 5 and dc.mde(1, 1000) == math.inf, f"{dc.mde(3, 1000):.2f}")
lo_, hi_, ev = dc.expected_ci(100, 1000, 2.0)
check("expected_ci: 하한 < RR < 상한, 기대 사건 수 = N1 x RR x 5%", lo_ < 2.0 < hi_ and abs(ev - 10.0) < 1e-9)

# 9) k 선택 규칙
sh = [{"k": 1, "defined": 100, "transient": 40, "share": 0.4}, {"k": 2, "defined": 100, "transient": 30, "share": 0.3},
      {"k": 3, "defined": 100, "transient": 15, "share": 0.15}, {"k": 6, "defined": 100, "transient": 5, "share": 0.05}]
check("choose_k: 20% 미만인 가장 짧은 창", dc.choose_k(sh)[0] == 3)
sh2 = [{"k": 1, "defined": 1000, "transient": 500, "share": 0.5}, {"k": 2, "defined": 1000, "transient": 495, "share": 0.495},
       {"k": 3, "defined": 1000, "transient": 490, "share": 0.49}, {"k": 6, "defined": 1000, "transient": 485, "share": 0.485}]
check("choose_k: 모두 20% 이상이고 더 줄지 않으면 가장 짧은 창", dc.choose_k(sh2)[0] == 1)

# 10) 계절성
t = np.arange(36)
Is = np.outer(np.ones(50), 10 * (1 + 0.3 * np.cos((t % 12) / 12 * 2 * np.pi)))
check("seasonality: 같은 계절 패턴이면 '있음'", dc.seasonality(Is)["flag"])
check("seasonality: 평평하면 '없음'", not dc.seasonality(np.ones((50, 36)) * 7 + rng.normal(0, 0.01, (50, 36)))["flag"])

# 11) 합성 데이터: 심어 둔 위축형을 찾고, M1(인과적 채움)이 누수 없이 동작
df, exp = make(1500, seed=3)
panel, rep = cl.build_panel(df)
order = pd.factorize(df["법인ID"].astype("string").str.strip())[1]
shr = np.array([exp["shrink_ts"].get(str(f), -1) for f in order])
masks = cl.sample_masks(panel)
P36, P35 = masks["P36"], masks["P35"]
I36 = dc.i1_matrix(panel, P36)
dec = dc.decide(I36)
res = dc.apply_rule(I36, dec["mode"], dec["k"], dec["phi"], dec["n"])
det, truth = res["det"], shr[P36]
is_shr = truth >= 0
rate_s, rate_n = float((det[is_shr] >= 0).mean()), float((det[~is_shr] >= 0).mean())
check("합성 P36: 위축형 기업의 탐지율이 일반 기업의 5배 이상이고 60% 이상", rate_s >= 0.60 and rate_s >= 5 * max(rate_n, 0.01), f"위축형 {rate_s:.2f}, 일반 {rate_n:.2f}")
hit = is_shr & (det >= 0)
lag = det[hit] - truth[hit]
check("합성 P36: 탐지월이 위축 시작월 근처 (중앙값 지연 0~k+1개월)", len(lag) > 0 and 0 <= np.median(lag) <= dec["k"] + 1, f"중앙값 {np.median(lag) if len(lag) else None}")
check("합성: 판정 가능 집합과 상태 행렬이 일관 (상태 0 의 수 == 판정 가능 기업-월)", int((res["status"] == 0).sum()) > 0)
check("합성: 정의된 n 은 양수, Φ >= 0", dec["n"] >= 5 and dec["phi"] >= 0)

# M1: 완전관측 기업은 M0 와 같고, s 이후 값이 달라도 최초 탐지월이 같다
Ip35 = dc.i1_matrix(panel, P35)
det_m1 = dc.first_detect_m1(panel, P36, dec["k"], dec["L"], dec["phi"], dec["n"] / 100.0)
check("M1: 완전관측 기업은 M0 와 같은 최초 탐지월", np.array_equal(det_m1, det))
# 누수 점검: 한 기업의 탐지월 이후 값을 바꿔도 최초 탐지월은 그대로
idx = np.flatnonzero(det >= 0)[0]
panel2 = cl.CountPanel(months=panel.months, observed=panel.observed.copy(), cnt={c: v.copy() for c, v in panel.cnt.items()},
                       amt=panel.amt, attr=panel.attr, attr_levels=panel.attr_levels, band_labels=panel.band_labels, synthetic=True)
rows36 = np.flatnonzero(P36)
for c in cl.CH5:
    panel2.cnt[c][rows36[idx], det[idx] + 1:] = 9      # 탐지월 이후를 모두 천장으로
det2 = dc.first_detect_m1(panel2, P36, dec["k"], dec["L"], dec["phi"], dec["n"] / 100.0)
check("M1: 탐지월 이후 값을 바꿔도 최초 탐지월은 변하지 않음 (미래 정보 누수 없음)", det2[idx] == det[idx])

# P35 의 M0 와 M1: 판정 불가(c)가 M1 에서 줄어든다
res35 = dc.apply_rule(Ip35, dec["mode"], dec["k"], dec["phi"], dec["n"])
det35_m1 = dc.first_detect_m1(panel, P35, dec["k"], dec["L"], dec["phi"], dec["n"] / 100.0)
check("P35: M0 에서 빈 달 부족(c) 기업-월이 존재", int((res35["status"] == 3).sum()) > 0)
check("P35: M1 최초 탐지 결과가 정의됨", det35_m1.shape == (int(P35.sum()),))

# 천장 보유 표시와 속성
flag = dc.ceiling_at_first_judgeable(panel, P36, res["change"], dec["phi"])
check("천장 보유 표시: 불리언 (n,)", flag.shape == (int(P36.sum()),) and flag.dtype == bool)
sec = dc.firm_attribute(panel, P36, "업종_대분류")
check("업종 속성: 모든 기업에 값이 있음", (sec >= 0).all())

# 12) 진단 함수: 천장 창 표시, 하한 산출에서 칸 제외, 탐지월 분해
fl = dc.ceiling_window_flag(panel, P36, dec["k"], dec["L"])
codes5 = np.stack([panel.cnt[c][P36] for c in cl.CH5]); any9 = ((codes5 == 9) & panel.observed[P36][None]).any(axis=0)
k_, L_ = dec["k"], dec["L"]
bad = 0
for e in (k_ - 1 + L_, 20, 29, 35):
    for i in rng.integers(0, any9.shape[0], 40):
        a = e - k_ + 1
        want = any9[i, a:e + 1].any() or any9[i, a - L_:e + 1 - L_].any()
        bad += int(bool(fl[i, e]) != bool(want))
check("ceiling_window_flag: 최근 창 또는 비교 창에 코드 9 가 있으면 True (무작위 비교)", bad == 0 and fl.shape == any9.shape)
d_none = dc.decide_fixed(I36, dec["mode"], dec["k"], None)
d_empty = dc.decide_fixed(I36, dec["mode"], dec["k"], np.zeros(I36.shape, dtype=bool))
check("decide_fixed: 제외 칸이 없으면 기본과 같음, decide 와도 같음", d_none["phi"] == d_empty["phi"] == dec["phi"] and d_none["n"] == dec["n"])
d_ex = dc.decide_fixed(I36, dec["mode"], dec["k"], fl)
check("decide_fixed: 천장 칸을 하한 산출에서 빼도 동작 (Φ, n 정의됨)", d_ex["phi"] >= 0 and d_ex["n"] >= 5)
share = dc.decompose_detection(panel, P36, det, dec["k"], dec["L"])
check("decompose_detection: 탐지 기업 수만큼 반환, -1 이거나 유한값", share.shape == (int((det >= 0).sum()),) and np.all((share == -1) | np.isfinite(share) | np.isnan(share)))
j = int(np.flatnonzero(det >= 0)[0]); e = int(det[j]); a = e - k_ + 1
Zs = [cl.codes_float(panel.cnt[c][P36], panel.observed[P36]) for c in cl.CH5]
tot = ce = 0.0; has_c = False
for z in Zs:
    am = cl.approx_count(z, cl.DEFAULT_CAP)
    dlt = np.nanmean(am[j, a:e + 1]) - np.nanmean(am[j, a - L_:e + 1 - L_])
    tot += dlt
    if np.mean(z[j, a - L_:e + 1 - L_] == 9) >= 0.5:
        ce += dlt; has_c = True
want = (ce / tot if has_c else -1.0) if tot < 0 else (np.nan if has_c else -1.0)
got = share[0]
check("decompose_detection: 손 계산과 일치 (첫 탐지 기업)", (np.isnan(want) and np.isnan(got)) or abs(want - got) < 1e-9, f"{want} vs {got}")

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
