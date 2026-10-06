"""types_cnt (P6 유형·자동이체·이동) 점검: 계획서 정의를 그대로 옮긴 느린 참조 구현과 대조.  python test_types_cnt.py

실제 데이터는 쓰지 않는다(합성 데이터). 참조 구현은 기업·층을 하나씩 도는 파이썬 반복문이며 types_cnt 의 함수를 호출하지 않는다.
"""
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402
import detect_cnt as dc  # noqa: E402
import types_cnt as tc  # noqa: E402
import validate_cnt as vc  # noqa: E402
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


# ======================================================================= 합성 패널
df, exp = make(3000, seed=7)
panel, rep = cl.build_panel(df)
masks = cl.sample_masks(panel)
P36, P35 = masks["P36"], masks["P35"]
MODE, K = "직전", 6
I1 = dc.i1_matrix(panel, P36)
FL = dc.ceiling_window_flag(panel, P36, K, dc.lag_for(MODE, K))
d_fix = dc.decide_fixed(I1, MODE, K, FL)
consts = vc.fit_constants(panel, P36, MODE, K, d_fix["phi"], d_fix["n"])
MIDS = list(cl.BAND_MIDS) + [cl.DEFAULT_CAP]


def approx(code):
    return MIDS[int(code)]


# ======================================================================= 1) 프로파일 · 오프라인 비중 · 유형
means, auto, nobs = tc.profile(panel, P36)
share = tc.offline_share(means)
obs36 = panel.observed[P36]
n_firm = obs36.shape[0]
bad = bad_a = bad_n = 0
for i in range(n_firm):
    ms = [t for t in range(0, 6) if obs36[i, t]]
    bad_n += int(nobs[i] != len(ms))
    m = {}
    for c in cl.CH5:
        v = [approx(panel.cnt[c][P36][i, t]) for t in ms if panel.cnt[c][P36][i, t] >= 0]
        m[c] = (sum(v) / len(v)) if (len(ms) >= 4 and v) else math.nan
    off = m[cl.OFFLINE[0]] + m[cl.OFFLINE[1]]
    tot = sum(m.values())
    want = off / tot if (np.isfinite(tot) and tot > 0) else math.nan
    bad += int(not ((math.isnan(want) and math.isnan(share[i])) or abs(want - share[i]) < 1e-12))
    va = [panel.cnt[cl.AUTO][P36][i, t] for t in ms if panel.cnt[cl.AUTO][P36][i, t] >= 0]
    wa = (sum(va) / len(va)) if (len(ms) >= 4 and va) else math.nan
    bad_a += int(not ((math.isnan(wa) and math.isnan(auto[i])) or abs(wa - auto[i]) < 1e-12))
check("프로파일 관측 달 수(2023.01~06) 일치", bad_n == 0)
check("오프라인 비중 = (창구+ATM)/국내 5채널 평균 근사 건수 합, 관측 달 < 4 또는 합 0 이면 미정(NaN) — 참조 구현과 일치", bad == 0, str(bad))
check("자동이체 평균 코드 일치", bad_a == 0, str(bad_a))
_cnt2 = {k: v.copy() for k, v in panel.cnt.items()}
for c in cl.CH5 + [cl.AUTO]:
    _cnt2[c][:, 6:] = (_cnt2[c][:, 6:] + 3) % 10      # 2023.07 이후 값을 모두 바꾼다
_panel2 = cl.CountPanel(months=panel.months, observed=panel.observed, cnt=_cnt2, amt=panel.amt, attr=panel.attr,
                        attr_levels=panel.attr_levels, band_labels=panel.band_labels, synthetic=True)
_m2, _a2, _n2 = tc.profile(_panel2, P36)
check("프로파일은 2023.01~06 만 사용: 2023.07 이후 값을 모두 바꿔도 오프라인 비중·자동이체 수준이 변하지 않음",
      np.allclose(share, tc.offline_share(_m2), equal_nan=True) and np.allclose(auto, _a2, equal_nan=True))

edges = tc.fit_type_edges(share)
typ = tc.assign_type(share, edges)
want_t = np.array([-1 if not np.isfinite(s) else (0 if s <= edges[0] + 1e-12 else 1 if s <= edges[1] + 1e-12 else 2) for s in share])
check("유형 배정 일치 (경계와 같으면 낮은 쪽)", (typ == want_t).all())
check("유형 크기: 세 유형 모두 존재하고 미정 아닌 기업이 대부분", set(np.unique(typ[typ >= 0])) <= {0, 1, 2} and (typ >= 0).mean() > 0.8, str(np.bincount(typ[typ >= 0])))
# 비중 0 이 많은 분포: 경계가 0 으로 겹쳐도 0 인 기업이 같은 유형에 모이고 혼합형이 빈다
sh0 = np.array([0.0] * 60 + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] + [np.nan] * 3)
e0 = tc.fit_type_edges(sh0)
t0 = tc.assign_type(sh0, e0)
check("동률: 비중 0 이 과반이면 경계1=경계2=0, 0 인 기업은 모두 낮은 유형(0), 나머지는 높은 유형(2), 혼합형(1) 비어 있음, NaN 은 -1",
      e0[0] == 0 and e0[1] == 0 and (t0[:60] == 0).all() and (t0[60:70] == 2).all() and (t0[70:] == -1).all() and not (t0 == 1).any(), str(e0))
ae = tc.fit_auto_edge(auto)
al = tc.assign_auto(auto, ae)
want_al = np.array([-1 if not np.isfinite(a) else int(a >= ae - 1e-12) for a in auto])
check("자동이체 높음/낮음 일치 (중앙값 이상 = 높음)", (al == want_al).all())
Tt = tc.make_types(panel, P36)
check("make_types: 경계를 주지 않으면 이 표본에서 정하고, 경계를 주면 그대로 씀 (P35만 적용)",
      np.allclose(Tt["type_edges"], edges) and np.allclose(tc.make_types(panel, P35, Tt["type_edges"], Tt["auto_edge"])["type_edges"], edges)
      and tc.make_types(panel, P35, Tt["type_edges"], Tt["auto_edge"])["auto_edge"] == Tt["auto_edge"])
chg = tc.type_change_rate(panel, P36, 100.0)
check("cap 100 으로 유형을 다시 만든 변경률이 0~1 사이", 0 <= chg <= 1, str(chg))

# ======================================================================= 2) C1 직접 표준화 탐지 비율
rng = np.random.default_rng(3)
Fn = 900
tp = rng.integers(-1, 3, Fn)
size = rng.integers(-1, 5, Fn)
jf = rng.random(Fn) < 0.8
det = rng.random(Fn) < (0.1 + 0.05 * np.maximum(size, 0))
sup = tc.std_support(tp, size, det, jf, min_n=10)
got = tc.std_rates(tp, size, det, jf, sup)
# 참조: 유형이 존재하는 모든 유형에서 기업 수 >= 10 인 분위만, 가중치 = 그 분위들의 전체(유형 정의) 판정 가능 기업 분포
cnt = defaultdict(int)
dcnt = defaultdict(int)
for i in range(Fn):
    if jf[i] and tp[i] >= 0 and size[i] >= 0:
        cnt[(tp[i], size[i])] += 1
        dcnt[(tp[i], size[i])] += int(det[i])
pres = [t for t in range(3) if any(cnt[(t, q)] > 0 for q in range(5))]
Qs = [q for q in range(5) if all(cnt[(t, q)] >= 10 for t in pres)]
tot = sum(sum(cnt[(t, q)] for t in range(3)) for q in Qs)
want = [sum((sum(cnt[(t2, q)] for t2 in range(3)) / tot) * (dcnt[(t, q)] / cnt[(t, q)]) for q in Qs) if t in pres else math.nan for t in range(3)]
check("직접 표준화 탐지 비율 일치 (고정 표준 인구 = 전체 규모 분포, 모든 유형에서 지원되는 분위만)", np.allclose(got, want, equal_nan=True), f"{got} vs {want}")
# 가중=복제: 기업 하나를 2번 복제한 것과 가중치 2 가 같다
w2 = np.ones(Fn)
w2[:50] = 2.0
rep_idx = np.concatenate([np.arange(Fn), np.arange(50)])
g_w = tc.std_rates(tp, size, det, jf, sup, w=w2)
g_r = tc.std_rates(tp[rep_idx], size[rep_idx], det[rep_idx], jf[rep_idx], sup)
check("가중 부트스트랩 = 기업 복제 (표준 인구 고정)", np.allclose(g_w, g_r, equal_nan=True))
bs = tc.boot_std_rates(tp, size, det, jf, B=200, seed=1)
check("boot_std_rates: 점추정이 std_rates 와 같고 (B,3) 배열 반환", np.allclose(bs["point"], got, equal_nan=True) and bs["boot"].shape == (200, 3))
rc = tc.ratio_ci(bs["boot"], 2, 0)
check("ratio_ci: 유효 반복 수와 범위 (lo <= hi)", rc["valid"] > 100 and rc["lo"] <= rc["hi"])

# ======================================================================= 3) 같은 집단 안 비교: 가중 RR 참조 구현
D = vc.prepare(panel, P36, consts)
U, F = D["units"], D["F"]


def naive_rr(u, w=None):
    w = np.ones(len(u)) if w is None else w
    cells = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for i in range(len(u)):
        c = cells[(u.e[i], u.size[i], u.sector[i])]
        if u.is_det[i]:
            c[0] += w[i]
            c[2] += w[i] * u.ev[i]
        else:
            c[1] += w[i]
            c[3] += w[i] * u.ev[i]
    groups = defaultdict(list)
    for key in cells:
        groups[(key[0], key[1])].append(key)
    obs = exp = 0.0
    for g, keys in groups.items():
        fb = any(cells[k][0] > 0 and cells[k][1] < vc.MIN_CTRL for k in keys)
        if not fb:
            for k in keys:
                n1, n0, e1, e0 = cells[k]
                if n1 > 0:
                    obs += e1
                    exp += n1 * e0 / n0 if n0 > 0 else 0.0
        else:
            n1 = sum(cells[k][0] for k in keys)
            n0 = sum(cells[k][1] for k in keys)
            if n1 > 0 and n0 >= vc.MIN_CTRL:
                obs += sum(cells[k][2] for k in keys)
                exp += n1 * sum(cells[k][3] for k in keys) / n0
    return obs / exp if exp > 0 else math.nan


typ_u = Tt["type"][U.firm]
aut_u = Tt["auto_level"][U.firm]
res_t = tc.rr_by_group(U, typ_u, [0, 1, 2], F, B=20, seed=11)
bad = 0
for gi, g in enumerate([0, 1, 2]):
    s = U.sub(typ_u == g)
    want = naive_rr(s) if (s.is_det.any() and (~s.is_det).any()) else math.nan
    pt = res_t["point"][gi]["rr"] if res_t["point"][gi] is not None else math.nan
    bad += int(not ((math.isnan(want) and math.isnan(pt)) or abs(want - pt) < 1e-9))
check("유형별 RR 점추정이 '같은 유형 안의 비탐지 기업과 비교'하는 참조 구현과 일치", bad == 0, str(res_t["point"]))
# 같은 재표집으로 계산한 반복값이 참조 구현과 같다 (재현: 같은 시드로 다중도 생성)
rng2 = np.random.default_rng(11)
bad = 0
for b in range(4):
    wf = vc._multiplicity(rng2, F)
    for gi, g in enumerate([0, 1, 2]):
        s = U.sub(typ_u == g)
        if not (s.is_det.any() and (~s.is_det).any()):
            continue
        want = naive_rr(s, wf[s.firm])
        got = res_t["boot"][b, gi]
        bad += int(not ((math.isnan(want) and math.isnan(got)) or abs(want - got) < 1e-9))
check("재표집 반복마다 위험집합·집단별 RR 을 다시 계산한 값이 참조 구현과 일치 (같은 기업 가중치를 모든 집단이 공유)", bad == 0)
lo_, hi_, nv = tc.group_ci(res_t, 0)
check("group_ci: 유효 반복이 있으면 범위 반환", (nv < 20) or lo_ <= hi_)
res_a = tc.rr_by_group(U, aut_u, [0, 1], F, B=20, seed=12)
check("자동이체 높음/낮음 RR 계산", len(res_a["point"]) == 2 and any(p is not None for p in res_a["point"]))

# ======================================================================= 4) C4 오프라인 비중 변화와 이동 신호
Dl = tc.offline_delta(panel, P36, D["er"])
ER = D["er"]
obs = panel.observed[P36]
Ys = {c: np.array([[approx(x) if x >= 0 and obs[i, t] else math.nan for t, x in enumerate(panel.cnt[c][P36][i])] for i in range(n_firm)]) for c in cl.CH5}


def naive_delta(i, e, need=5):
    pairs = [(e - 5 + j, e - 11 + j) for j in range(6)]
    pairs = [(a, b) for a, b in pairs if all(not math.isnan(Ys[c][i, a]) and not math.isnan(Ys[c][i, b]) for c in cl.CH5)]
    if len(pairs) < need:
        return math.nan
    cur = {c: sum(Ys[c][i, a] for a, b in pairs) / len(pairs) for c in cl.CH5}
    old = {c: sum(Ys[c][i, b] for a, b in pairs) / len(pairs) for c in cl.CH5}
    tc_, to_ = sum(cur.values()), sum(old.values())
    if tc_ <= 0 or to_ <= 0:
        return math.nan
    return (cur[cl.OFFLINE[0]] + cur[cl.OFFLINE[1]]) / tc_ - (old[cl.OFFLINE[0]] + old[cl.OFFLINE[1]]) / to_


sample_i = rng.choice(n_firm, 120, replace=False)
bad = 0
for e in (17, 21, 25, 29):
    for i in sample_i:
        want = naive_delta(i, e)
        bad += int(not ((math.isnan(want) and math.isnan(Dl[i, e])) or abs(want - Dl[i, e]) < 1e-9))
check("오프라인 비중 변화 Δ(i,e) = 최근 6개월 비중 - 직전 6개월 비중 (쌍 맞춤) 참조 구현과 일치", bad == 0, str(bad))
check("Δ 는 판정 기준월 밖에서 정의되지 않음", np.isnan(Dl[:, :17]).all())

# 심은 이동: 일부 기업에서 2024.09(인덱스 20) 이후 창구·ATM 을 줄이고 인터넷·스마트를 늘려 총량은 비슷하게 유지
pl_cnt = {k: v.copy() for k, v in panel.cnt.items()}
rng3 = np.random.default_rng(5)
idx36 = np.flatnonzero(P36)
planted_local = rng3.choice(len(idx36), 250, replace=False)
for li in planted_local:
    gi_ = idx36[li]
    for t in range(20, 36):
        for c in cl.OFFLINE:
            pl_cnt[c][gi_, t] = max(int(pl_cnt[c][gi_, t]) - 3, 0)
        for c in (cl.CH5[1], cl.CH5[2]):
            pl_cnt[c][gi_, t] = min(int(pl_cnt[c][gi_, t]) + 3, 9)
panel_pl = cl.CountPanel(months=panel.months, observed=panel.observed, cnt=pl_cnt, amt=panel.amt, attr=panel.attr,
                         attr_levels=panel.attr_levels, band_labels=panel.band_labels, synthetic=True)
Dp = vc.prepare(panel_pl, P36, consts)
Dlp = tc.offline_delta(panel_pl, P36, Dp["er"])
for kept in (True, False):
    Us_, first_ = tc.build_shift_units(Dp, Dlp, consts["theta_o1"], activity_kept=kept)
    st_, ch_, det_ = Dp["status"], Dp["change"], Dp["det"]
    already = lambda i, e: det_[i] >= 0 and e >= det_[i]  # noqa: E731
    npct = Dp["n_pct"] / 100.0

    def cond(i, e):
        if st_[i, e] != 0 or already(i, e) or not (Dlp[i, e] <= tc.SHIFT_DELTA + 1e-9):
            return False
        r = ch_.r[i, e]
        return (abs(r) <= tc.SHIFT_R + 1e-9) if kept else (r > -npct + 1e-9)

    first_n = {}
    for i in range(n_firm):
        for e in Dp["er"]:
            if cond(i, e):
                first_n[i] = e
                break
    ev_set = {(i, e) for i, e in first_n.items() if not np.isnan(Dp["G"][i, e])}
    got_ev = {(int(f), int(e)) for f, e, d in zip(Us_.firm, Us_.e, Us_.is_det) if d}
    ctrl_n = set()
    for e in Dp["er"]:
        for i in range(n_firm):
            if st_[i, e] == 0 and not already(i, e) and not np.isnan(Dp["G"][i, e]) and (i not in first_n or first_n[i] > e):
                ctrl_n.add((i, e))
    got_ct = {(int(f), int(e)) for f, e, d in zip(Us_.firm, Us_.e, Us_.is_det) if not d}
    check(f"이동 단위(최초 이동월, 총 활동 조건 {'있음' if kept else '없음'})와 위험집합(그 달 아직 이동·위축 탐지 안 한 판정 가능 기업)이 참조 구현과 일치",
          ev_set == got_ev and ctrl_n == got_ct, f"{len(ev_set)} vs {len(got_ev)}, {len(ctrl_n)} vs {len(got_ct)}")
    if kept:
        n_pl = sum(1 for (f, e) in got_ev if f in set(idx36[planted_local].tolist()) or True)
        print(f"    (합성) 이동 기업 {len(got_ev)}곳")
Uk, fk = tc.build_shift_units(Dp, Dlp, consts["theta_o1"], True)
planted_set = set(planted_local.tolist())
ev_firms = [int(f) for f, d in zip(Uk.firm, Uk.is_det) if d]
share_pl = np.mean([f in planted_set for f in ev_firms]) if ev_firms else 0
check("합성: 심은 이동(창구·ATM↓ 인터넷·스마트↑, 총량 유지) 기업이 이동 신호의 대부분", len(ev_firms) >= 30 and share_pl > 0.5, f"{len(ev_firms)}곳, 심은 비율 {share_pl:.2f}")
check("이동 기업은 위축 탐지월 이전에만 이동으로 잡힘 (이미 위축 탐지된 달 이후는 제외)",
      all(not (Dp["det"][int(f)] >= 0 and int(e) >= Dp["det"][int(f)]) for f, e, d in zip(Uk.firm, Uk.e, Uk.is_det) if d))
# 이동과 위축 탐지는 같은 칸에서 동시에 성립할 수 없음 (총 활동 ±20% vs -35% 이하)
both = [(int(f), int(e)) for f, e, d in zip(Uk.firm, Uk.e, Uk.is_det) if d and Dp["change"].r[int(f), int(e)] <= -Dp["n_pct"] / 100.0 + 1e-9]
check("총 활동 유지 조건의 이동 칸에는 위축 탐지 문턱 이하의 칸이 없음", not both)
Ra = tc.analyze_shift(Dp, Dlp, consts, Dp["F"], B=30, seed=3, min_units=5, min_events=1)
check("analyze_shift: 이동 기업이 충분하면 RR 계산, 기술 전용 플래그 없음", (not Ra["descriptive"]) and "R" in Ra and np.isfinite(Ra["R"]["rr"]))
Rb = tc.analyze_shift(Dp, Dlp, consts, Dp["F"], B=30, seed=3, min_units=10 ** 6)
check("analyze_shift: 이동 기업 수 기준 미달이면 기술만 (RR 계산 안 함)", Rb["descriptive"] and "R" not in Rb)

# ======================================================================= 5) 전체 실행 (P36 · P35만에 같은 코드, 경계는 P36 고정값)
T36 = tc.make_types(panel, P36)
out36 = tc.run_all(panel, P36, D, T36, consts, B=60, seed0=100)
check("run_all(P36): C1~C4 결과가 모두 만들어짐", all(k in out36 for k in ("c1", "c2", "c3", "c4_kept", "c4_any")) and out36["c1"]["boot"].shape == (60, 3))
D35 = vc.prepare(panel, P35, consts)
T35 = tc.make_types(panel, P35, T36["type_edges"], T36["auto_edge"])
out35 = tc.run_all(panel, P35, D35, T35, consts, B=60, seed0=200)
check("run_all(P35만): P36 경계를 그대로 써서 계산", np.allclose(T35["type_edges"], T36["type_edges"]) and out35["c2"]["boot"].shape == (60, 3))
print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
