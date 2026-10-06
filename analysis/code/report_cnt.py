"""보고/종합분석보고서.ipynb(원본 실행 버전)용 도구.

1. run_pipeline  : 노트북 01~08 의 코드 셀을 순서대로 실행한다 (원본 -> 패널 -> 신호 -> 입금 검증 -> ... -> 보강 점검).
                   노트북과 똑같은 코드·같은 난수 시드로 돌리므로 사용자가 이미 돌린 결과와 같은 값이 나온다.
                   이미 끝난 단계는 건너뛸 수 있다 (결과 파일이 있을 때).
2. load_results  : 01~08 이 저장한 '집계 결과'(CSV·JSON) 만 읽어 보고서의 그림·표에 쓸 값으로 바꾼다.
                   행 단위 파일(npz)은 열지 않는다.
3. checkpoints   : 이번 실행 값을 이 분석의 기준 실행(2026-10-02) 값과 비교한 점검표.

레포 README '데이터 취급 원칙'을 그대로 따른다: 화면에는 집계값만, 법인ID 는 어디에도 남기지 않는다.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
NB_DIR = HERE.parent / "notebooks"

# (단계, 노트북 파일, 끝났다는 표시 파일(interim 아래))
STEPS = [
    ("01", "01_표본정의_패널구축.ipynb", "nb01_summary.json"),
    ("02", "02_건수수치화.ipynb", "quant_spec.json"),
    ("03", "03_채널구조탐색.ipynb", "nb03_summary.json"),
    ("04", "04_활동위축_탐지규칙.ipynb", "detect_spec_v2.json"),
    ("05", "05_입금검증.ipynb", "nb05_summary.json"),
    ("06", "06_채널유형_위축비교.ipynb", "nb06_summary.json"),
    ("07", "07_민감도.ipynb", "nb07_summary.json"),
    ("08", "08_보강검증.ipynb", "nb08_summary.json"),
]
# 보고서 노트북은 01~08 만 돌린다. 09(P10 추가 민감도)는 run_all.py 가 이어서 돌리거나 steps 에 "09" 를 직접 지정한다.
EXTRA_STEPS = [
    ("09", "09_추가민감도_후보월과대칭결과.ipynb", "nb09_summary.json"),
]
ALL_STEPS = STEPS + EXTRA_STEPS


# --------------------------------------------------------------------------- 실행기
def notebook_script(nb_path: Path) -> str:
    """노트북의 코드 셀만 이어 붙인 스크립트. 05 의 35개월 법인 잠금은 환경변수로 연다.

    (35개월 법인의 입금 결과는 P36 분석·기록이 끝난 뒤에만 여는 사전 규칙이 있다. 이 보고서는 그 단계를 지난 뒤의 종합본이라 연다.)"""
    nb = json.loads(Path(nb_path).read_text(encoding="utf-8"))
    text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    return re.sub(r"HOLDOUT_OPEN = (True|False)\n", 'HOLDOUT_OPEN = os.environ.get("CNT_HOLDOUT_OPEN") == "1"\n', text)


def step_done(work: Path, step: tuple) -> bool:
    """끝난 단계인가. 05 는 35개월 법인 결과까지 열린 요약이어야 한다."""
    p = Path(work) / "interim" / step[2]
    if not p.exists():
        return False
    if step[0] == "05":
        try:
            return bool(json.loads(p.read_text(encoding="utf-8")).get("holdout_open"))
        except (OSError, ValueError):
            return False
    return True


def run_pipeline(data_path, work: Path, *, sheet: str = "all", boot: int | None = None, force: bool = False, steps=None,
                 expect_real_counts: bool = True, log_dir: Path | None = None, python: str | None = None, verbose: bool = True) -> list[dict]:
    """01~08 을 차례로 실행한다(steps 에 "09" 를 넣으면 P10 추가 민감도도). 반환: 단계별 {step, status, seconds}. 실패하면 그 단계에서 멈추고 예외를 낸다.

    force=False 이면 끝난 단계(결과 파일이 있는 단계)는 건너뛴다. boot 는 부트스트랩 횟수(없으면 노트북 기본값: 03 은 10,000, 나머지 2,000).
    화면에는 단계 이름과 걸린 시간만 낸다 (노트북의 표 출력은 로그 파일 work/logs 에 저장 -- 집계값뿐이고 .gitignore 대상 폴더)."""
    work = Path(work)
    log_dir = Path(log_dir) if log_dir else work / "logs"
    env = {**os.environ, "CNT_DATA_PATH": str(data_path), "CNT_SHEET": str(sheet), "CNT_WORK_DIR": str(work), "MPLBACKEND": "Agg",
           "PYTHONIOENCODING": "utf-8", "CNT_HOLDOUT_OPEN": "1"}
    if not expect_real_counts:
        env["CNT_NO_EXPECT"] = "1"
    if boot:
        env["CNT_BOOT"] = str(int(boot))
    want = [s for s in ALL_STEPS if (s in STEPS if steps is None else s[0] in {str(x) for x in steps})]
    result = []
    for st in want:
        name = st[1].replace(".ipynb", "")
        if not force and step_done(work, st):
            if verbose:
                print(f"[{st[0]}] 건너뜀 — 이미 계산된 결과 사용 ({name}).  처음부터 다시 계산하려면 FORCE_RERUN = True")
            result.append({"step": st[0], "status": "skipped", "seconds": 0.0})
            continue
        script = notebook_script(NB_DIR / st[1])
        log_dir.mkdir(parents=True, exist_ok=True)
        script_path = log_dir / f"_run_{st[0]}.py"
        script_path.write_text(script, encoding="utf-8")
        t0 = time.time()
        if verbose:
            print(f"[{st[0]}] 실행 중 — {name} ...", flush=True)
        r = subprocess.run([python or sys.executable, str(script_path)], cwd=str(NB_DIR), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        sec = time.time() - t0
        (log_dir / f"{st[0]}.log").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
        script_path.unlink(missing_ok=True)
        if r.returncode != 0:
            tail = (r.stdout[-1200:] + "\n" + r.stderr[-1500:]).strip()
            raise RuntimeError(f"단계 {st[0]} ({name}) 실행 실패. 로그: {log_dir / (st[0] + '.log')}\n에러 종류와 줄 번호만 확인하세요 (데이터 값이 섞인 메시지는 공유 금지):\n{tail}")
        if verbose:
            print(f"      완료 ({sec:.0f}초)")
        result.append({"step": st[0], "status": "done", "seconds": sec})
    return result


# --------------------------------------------------------------------------- 집계 결과 불러오기
def _num(x):
    if x is None:
        return None
    x = str(x).strip().replace(",", "")
    if x in ("", "-", "nan") or x.startswith("<"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _int(x, default: int = 0) -> int:
    """가려진 칸('<5')·빈 칸은 default(0) 로 읽는다 (표본이 작은 자료에서도 불러오기가 멈추지 않게)."""
    v = _num(x)
    return default if v is None else int(v)


def _ci_text(s):
    """'4.69 (3.75~5.95)' -> (4.69, 3.75, 5.95), 해석할 수 없으면 None"""
    m = re.match(r"\s*([-\d.]+)\s*\(([-\d.]+)\s*~\s*([-\d.]+)\)", str(s))
    return tuple(float(m.group(i)) for i in (1, 2, 3)) if m else None


def _range(s):
    """'1.247 ~ 36.656' -> [1.247, 36.656]"""
    return [_num(x) for x in str(s).replace(" ", "").split("~")]


def _jl(name, obj):
    """파이썬 리터럴 한 줄 (NaN·무한대도 읽히게)"""
    txt = json.dumps(obj, ensure_ascii=False, allow_nan=True)
    for a, b in (("NaN", "float('nan')"), ("-Infinity", "float('-inf')"), ("Infinity", "float('inf')"), ("null", "None"), ("true", "True"), ("false", "False")):
        txt = re.sub(rf"(?<![\w'\"]){re.escape(a)}(?![\w'\"])", b, txt)
    return f"{name} = {txt}\n"


_DICT_NAMES = {"N_SAMPLE", "OBS_MONTHS", "MISSING_KIND", "FILL_BY_VAR", "FILL_ALL", "USAGE_P36", "CEIL_CHANNELS", "SPEARMAN", "YOY_CODE_CHANGE", "TRANSITION", "NB03",
               "USAGE_BY_YEAR", "FX_BY_SECTOR", "RULE", "TRANSIENT_30", "IQR_BY_ACT", "V2_COUNTS", "PERIOD", "FLOOR_V1", "POWER_V2", "MAIN", "FLOOR_EXCLUDED",
               "JUDGEABLE_FIRMS", "CONSTANTS", "SHIFT", "VARIANT_BASE", "THRESHOLD_OVERLAP", "TWO_BY_TWO", "HIT_P36", "HIT_P35", "TIMING", "YOY"}


def build_text(work) -> str:
    """집계 결과(CSV·JSON)를 읽어 보고서 상수(파이썬 코드 문자열)를 만든다. 행 단위 파일은 열지 않는다."""
    work = Path(work)
    OUT, INT = work / "output", work / "interim"

    def rc(rel):
        """집계 CSV 를 읽는다. 파일이 없거나 비어 있으면(표본이 작아 표가 만들어지지 않은 경우 등) 빈 표를 주고 경고에 남긴다."""
        try:
            return pd.read_csv(OUT / rel, encoding="utf-8-sig", thousands=",", dtype=str)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            warns.append(f"{rel}: 파일 없음/비어 있음")
            return pd.DataFrame(columns=[f"c{i}" for i in range(12)])

    def js(name):
        return json.loads((INT / name).read_text(encoding="utf-8"))

    L, warns = [], []

    def put(name, fn, default=None):
        """상수 하나를 만든다. 표가 비었거나 열이 달라 실패하면 그 상수만 빈 값으로 두고 경고에 남긴다 (불러오기 전체가 멈추지 않게)."""
        try:
            L.append(_jl(name, fn()))
        except Exception as e:  # noqa: BLE001
            warns.append(f"{name}: {type(e).__name__}")
            L.append(_jl(name, {} if name in _DICT_NAMES else [] if default is None else default))

    j5, j6, j7, j8, j3, j1 = js("nb05_summary.json"), js("nb06_summary.json"), js("nb07_summary.json"), js("nb08_summary.json"), js("nb03_summary.json"), js("nb01_summary.json")

    # ---- P1
    val = rc("nb01/validation.csv")
    put("N_ALL", lambda: _int(val.iloc[1]["실제"]))
    put("N_ROWS", lambda: _int(val.iloc[0]["실제"]))
    smp = rc("nb01/samples.csv")
    put("N_SAMPLE", lambda: {"P36": _int(smp.iloc[0, 1]), "P35만": _int(smp.iloc[1, 1]), "P35+": _int(smp.iloc[2, 1])})
    om = rc("nb01/obs_months_30_36.csv")
    put("OBS_MONTHS", lambda: {int(a): _int(b) for a, b in zip(om.iloc[:, 0], om.iloc[:, 1])})
    mk = rc("nb01/p35_missing_kind.csv")
    put("MISSING_KIND", lambda: {r.iloc[0]: [_int(r.iloc[1]), _num(r.iloc[2])] for _, r in mk.iterrows()})
    fv = rc("nb01/fill_accuracy_by_var.csv")
    put("FILL_BY_VAR", lambda: {r.iloc[0]: [_num(r.iloc[1]), _num(r.iloc[2])] for _, r in fv.iterrows()})
    put("FILL_ALL", lambda: {"exact": round(j1["fill_exact_pct"], 1), "within1": round(j1["fill_within1_pct"], 1)})

    # ---- P2
    bd = rc("nb02/bands.csv")
    put("BANDS", lambda: [[_int(r.iloc[0]), r.iloc[1], r.iloc[2]] for _, r in bd.iterrows()])
    uc = rc("nb02/usage_ceiling.csv")
    uc = uc[uc.iloc[:, 0] == "P36"]
    put("USAGE_P36", lambda: {r.iloc[1]: [_num(r.iloc[3]), _num(r.iloc[5])] for _, r in uc.iterrows()})
    cs = rc("nb02/i1_cap_sensitivity.csv")
    put("CEIL_CHANNELS", lambda: {"0개": _num(cs.iloc[0, 1]), "1개": _num(cs.iloc[1, 1]), "2개 이상": _num(cs.iloc[2, 1])})
    ib = rc("nb02/i1_bounds.csv")
    put("SPEARMAN", lambda: {"I1(cap60) vs 하한 합": _num(ib.iloc[5, 1]), "I1(cap60) vs 상한 합": _num(ib.iloc[6, 1]), "하한 합 vs 상한 합": _num(ib.iloc[7, 1]),
                              "I1(cap60) vs 순서 코드 합": _num(cs.iloc[8, 1])})
    yc = rc("nb02/yoy_code_change.csv")
    put("YOY_CODE_CHANGE", lambda: {r.iloc[0]: [_num(x) for x in r.iloc[2:9]] for _, r in yc.iterrows()})

    # ---- P3
    put("CH8", lambda: ["창구", "인터넷", "스마트", "폰", "ATM", "자동이체", "외환수출", "외환수입"])
    bc, wc = rc("nb03/between_corr_P36.csv"), rc("nb03/within_corr_P36.csv")
    put("CORR_BETWEEN", lambda: [[_num(x) for x in bc.iloc[i, 1:9]] for i in range(8)])
    put("CORR_WITHIN", lambda: [[_num(x) for x in wc.iloc[i, 1:9]] for i in range(8)])
    tr = rc("nb03/transition_P36.csv")
    put("TRANSITION_COLS", lambda: list(tr.columns[1:]))
    put("TRANSITION", lambda: {r.iloc[0]: [_num(x) for x in r.iloc[1:5]] for _, r in tr.iterrows()})
    pv = rc("nb03/pca_variance.csv")
    put("PCA_VAR", lambda: [_num(x) for x in pv.iloc[:, 2]])
    put("NB03", lambda: j3)
    ua = rc("nb03/usage_by_year.csv")
    put("USAGE_BY_YEAR", lambda: {r.iloc[0]: [_num(x) for x in r.iloc[1:4]] for _, r in ua.iterrows()})
    fx = rc("nb03/fx_by_sector.csv")
    put("FX_BY_SECTOR", lambda: {r.iloc[0]: [_int(r.iloc[1]), _num(r.iloc[2])] for _, r in fx.iterrows()})

    # ---- P4
    spec = js("detect_spec_v2.json")
    rule = spec["rule"]
    put("RULE", lambda: {"k": rule["k"], "phi": round(rule["phi"], 2), "n_pct": rule["n_pct"], "q05": round(rule["q05"] * 100, 1),
                          "theta_o1": round(j5["constants"]["theta_o1"] * 100, 1), "edges": [round(x, 1) for x in j5["constants"]["edges"]]})
    ts = rc("nb04/transient_share.csv")
    def _transient():
        out = {}
        for _, r in ts.iterrows():
            out.setdefault(r.iloc[0], []).append([_int(r.iloc[2]), _num(r.iloc[5])])
        return out

    put("TRANSIENT_30", _transient)
    s35 = rc("nb07/selection_shares.csv")
    s35 = s35[s35.iloc[:, 0] == "35%"]
    put("TRANSIENT_35_P36", lambda: [[_int(r.iloc[1]), _num(r.iloc[3])] for _, r in s35.iterrows()])
    dq = rc("nb04/diag_ceiling_iqr.csv")
    put("IQR_BY_ACT", lambda: {g: [[r.iloc[2], _num(r.iloc[5]), _num(r.iloc[6])] for _, r in dq[dq.iloc[:, 0] == g].iterrows()] for g in dq.iloc[:, 0].unique()})
    v2 = rc("nb04/v2_counts.csv")
    put("V2_COUNTS", lambda: {r.iloc[0]: {"phi": _num(r.iloc[1]), "n": _num(r.iloc[2]), "judgeable": _int(r.iloc[3]), "judge_pct": _num(r.iloc[4]),
                                           "ceiling_pct": _num(r.iloc[5]), "detected": _int(r.iloc[6]), "rate_pct": _num(r.iloc[7]), "mde": _num(r.iloc[8])} for _, r in v2.iterrows()})
    ps = rc("nb07/period_availability.csv")
    put("PERIOD", lambda: {r.iloc[0]: {"months": _int(r.iloc[1]), "judge": _int(r.iloc[2]), "det": _int(r.iloc[3]), "ctrl": _int(r.iloc[6])} for _, r in ps.iterrows()})
    fs = rc("nb04/floor_scope.csv")
    put("FLOOR_V1", lambda: {r.iloc[0]: _int(r.iloc[1]) for _, r in fs.iterrows()})
    ub = rc("nb05/units_by_month.csv")
    put("UNITS_BY_MONTH", lambda: [[_int(r.iloc[0]), _int(r.iloc[1]), _int(r.iloc[2])] for _, r in ub.iterrows()])
    put("POWER_V2", lambda: {k: {"N1": v["N1"], "N0": v["N0"], "mde": round(v["mde"], 2)} for k, v in spec["power"].items()})

    # ---- P5
    m = j5["P36"]["main"]
    put("MAIN", lambda: {"n1": int(m["n1"]), "obs": int(m["obs"]), "exp": round(m["exp"], 1), "p_det": round(m["p_det"] * 100, 1), "p_ctrl": round(m["p_ctrl"] * 100, 1),
                          "rr": round(m["rr"], 2), "lo": round(m["lo"], 2), "hi": round(m["hi"], 2), "p_one": round(m["p_one"], 4), "B": int(m["B_valid"]), "n0": int(m["n0"])})
    hs = j5["holdout"]
    rows = [["36개월 법인 (규칙 결정)", int(m["n1"]), m["rr"], m["lo"], m["hi"], m["p_det"] * 100, m["p_ctrl"] * 100]]
    for key, lab in (("P35만", "35개월 법인만 (독립 검증)"), ("P35+", "35개월 이상 전체 (표본 확대)")):
        x = hs[key]["main"]
        rows.append([lab, int(x["n1"]), x["rr"], x["lo"], x["hi"], x["p_det"] * 100, x["p_ctrl"] * 100])
    put("SAMPLE_RR", lambda: [[r[0], r[1]] + [round(v, 2) for v in r[2:5]] + [round(v, 1) for v in r[5:]] for r in rows])
    het = rc("nb05/heterogeneity.csv")
    h5 = het.iloc[:5]
    put("SIZE_RR", lambda: [[r.iloc[0], _int(r.iloc[2]), _int(r.iloc[7]), _num(r.iloc[3]), _ci_text(r.iloc[8])] for _, r in h5.iterrows()])
    put("SIZE_JUDGEABLE", lambda: [_int(r.iloc[1]) for _, r in h5.iterrows()])
    ce = het.iloc[5:7]
    put("CEIL_RR", lambda: [[r.iloc[0], _int(r.iloc[2]), _int(r.iloc[7]), _ci_text(r.iloc[8])] for _, r in ce.iterrows()])
    lm = rc("nb05/leave_one_month.csv")
    put("LOMO", lambda: [[_int(r.iloc[0]), _int(r.iloc[1]), _num(r.iloc[2])] for _, r in lm.iterrows()])
    bl = rc("nb05/balance.csv")
    put("BALANCE", lambda: [[r.iloc[0], _num(r.iloc[1]), _num(r.iloc[2]), _num(r.iloc[3]), _num(r.iloc[4]), _num(r.iloc[5])] for _, r in bl.iterrows()])
    hp = rc("nb05/hc2_path.csv")
    put("PATH", lambda: [[_int(r.iloc[0]), r.iloc[1], _num(r.iloc[5]), _range(r.iloc[6])] for _, r in hp.iterrows()])
    hsum = rc("nb05/hc2_summary.csv")
    put("PATH_SUMMARY", lambda: [[r.iloc[0], _num(r.iloc[1]), _range(r.iloc[2])] for _, r in hsum.iterrows()])
    fe = rc("nb05/floor_excluded.csv")
    put("FLOOR_EXCLUDED", lambda: {"n": _int(fe.iloc[0, 1]), "i1_median": _num(fe.iloc[0, 2]), "weak_pct": _num(fe.iloc[0, 3])})
    put("JUDGEABLE_FIRMS", lambda: {"n": _int(fe.iloc[1, 1]), "i1_median": _num(fe.iloc[1, 2]), "weak_pct": _num(fe.iloc[1, 4])})
    ex = rc("nb05/explore_surge.csv")
    put("SURGE", lambda: [[r.iloc[0], _int(r.iloc[1]), _num(r.iloc[4]), _num(r.iloc[5]), _ci_text(r.iloc[7])] for _, r in ex.iterrows()])
    ud = rc("nb05/explore_updown.csv")
    put("UPDOWN", lambda: [[r.iloc[0], _int(r.iloc[1]), _num(r.iloc[2]), _num(r.iloc[3]), _ci_text(r.iloc[5])] for _, r in ud.iterrows()])
    aux = rc("nb05/aux_o2_o3.csv")
    put("AUX", lambda: [[r.iloc[0], r.iloc[1]] for _, r in aux.iterrows()])
    put("RR_O3", lambda: _ci_text(aux.iloc[0, 1]))
    s1 = rc("nb05/s1_and_normal.csv")
    put("S1", lambda: [[r.iloc[0], _int(r.iloc[1]), _ci_text(r.iloc[2])] for _, r in s1.iterrows()])
    ng = rc("nb05/n_grid.csv")
    put("N_GRID_NB05", lambda: [[_num(r.iloc[0]), _int(r.iloc[1]), _num(r.iloc[2]), _num(r.iloc[3]), _num(r.iloc[4]), _num(r.iloc[6])] for _, r in ng.iterrows()])
    nst = rc("nb05/n_stability.csv")
    put("N_STABILITY", lambda: [[r.iloc[0], _num(r.iloc[3])] for _, r in nst.iterrows()])
    put("CONSTANTS", lambda: {"theta_o1_pct": round(j5["constants"]["theta_o1"] * 100, 1), "theta_o3_pct": round(j5["constants"]["theta_o3"] * 100, 1),
                               "edges": [round(x, 1) for x in j5["constants"]["edges"]]})

    # ---- P6
    gb = rc("nb06/groups_both.csv")
    g36 = gb[gb.iloc[:, 0] == "P36"]
    put("GROUP_RR", lambda: [[r.iloc[1], _int(r.iloc[2]), _ci_text(r.iloc[3])] for _, r in g36.iterrows() if _ci_text(r.iloc[3])])
    g35 = gb[gb.iloc[:, 0] == "P35만"]
    put("GROUP_RR_P35", lambda: [[r.iloc[1], _int(r.iloc[2]), _ci_text(r.iloc[3])] for _, r in g35.iterrows() if _ci_text(r.iloc[3])])
    ov = rc("nb06/overall.csv")
    put("RATIOS", lambda: [[r.iloc[0], _ci_text(r.iloc[1]), _ci_text(r.iloc[2])] for _, r in ov.iterrows() if _ci_text(r.iloc[1])])
    c1 = rc("nb06/c1_standardized.csv")
    put("TYPE_STD_RATE", lambda: [[r.iloc[0], _num(r.iloc[1]), _range(r.iloc[2]), _num(r.iloc[3])] for _, r in c1.iterrows()])
    c3 = rc("nb06/c3_auto_rr.csv")
    put("AUTO_RATES", lambda: [[r.iloc[0], _int(r.iloc[1]), _num(r.iloc[3]), _num(r.iloc[4])] for _, r in c3.iterrows()])
    c2 = rc("nb06/c2_type_rr.csv")
    put("TYPE_RATES", lambda: [[r.iloc[0], _int(r.iloc[1]), _num(r.iloc[3]), _num(r.iloc[4])] for _, r in c2.iterrows()])
    tc = rc("nb06/type_context.csv")
    put("TYPE_CONTEXT", lambda: [[r.iloc[0], _int(r.iloc[1]), _num(r.iloc[4]), _num(r.iloc[6]), _num(r.iloc[7])] for _, r in tc.iterrows()])
    n_shift = j6["P36"]["c4_kept"]["n_units"]
    put("SHIFT", lambda: {"n_units": n_shift, "share_judgeable_pct": round(n_shift / _num(v2.iloc[0, 3]) * 100, 1)})

    # ---- P7
    V = []
    for name, v in j7["variants"].items():
        grp, lab = [x.strip() for x in name.split("|", 1)]
        V.append([grp, lab, round(v["rr"], 2), round(v["lo"], 2), round(v["hi"], 2), v["n_det"], None if v["jaccard"] is None else round(v["jaccard"], 2)])
    put("VARIANTS", lambda: V)
    b = j7["base"]
    put("VARIANT_BASE", lambda: {"rr": round(b["rr"], 2), "lo": round(b["lo"], 2), "hi": round(b["hi"], 2)})
    put("THRESHOLD_OVERLAP", lambda: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in j7["threshold_overlap"].items()})
    put("SELECTION_SAME", lambda: j7["selection"]["same_rule"])

    # ---- P9
    A = j8["A"]
    put("TWO_BY_TWO", lambda: {k: A["two_by_two"]["P36"][k] for k in ("a", "b", "c", "d")})
    for key, name in (("P36", "HIT_P36"), ("P35", "HIT_P35")):
        put(name, lambda: {"precision": round(A["two_by_two"][key]["precision"] * 100, 1), "recall": round(A["two_by_two"][key]["recall"] * 100, 1),
                            "ctrl_raw": round(A["two_by_two"][key]["ctrl_rate"] * 100, 1), "ctrl_std": round(A["std_ctrl"][key] * 100, 1)})
    put("TIMING", lambda: {k: {kk: v[kk] for kk in ("n_judgeable", "n_weak", "early", "late", "none")} for k, v in A["firm_timing"].items()})
    keep = ("n", "a", "b", "c", "d", "n_signal", "n_detected", "precision", "recall", "rr", "p_ctrl_std", "n_months")
    put("N_GRID", lambda: [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in g.items() if k in keep} for g in A["grid"]])
    B = j8["B"]

    def pack(d):
        return {"rr": round(d["rr"], 2), "lo": round(d["lo"], 2), "hi": round(d["hi"], 2), "n": d["n_det"], "p_det": round(d["p_det"] * 100, 1), "p_ctrl": round(d["p_ctrl"] * 100, 1)}

    put("YOY", lambda: {"P36_base": pack(B["P36_base"]), "P36_yoy": pack(B["P36_yoy"]), "P35_base": pack(B["P35_base"]), "P35_yoy": pack(B["P35_yoy"]),
                         "ratio_P36": [round(x, 2) for x in B["P36_yoy"]["ratio"]], "ratio_P35": [round(x, 2) for x in B["P35_yoy"]["ratio"]],
                         "jaccard_P36": round(B["P36_yoy"]["jaccard"], 2), "jaccard_P35": round(B["P35_yoy"]["jaccard"], 2), "rule": B["yoy_rule"], "overlap": B["overlap"]})
    hit2 = rc("nb08/two_rules_hit.csv")
    put("TWO_RULES_HIT", lambda: [[r.iloc[0], _num(r.iloc[1]), _num(r.iloc[3]), _num(r.iloc[4])] for _, r in hit2.iterrows()])
    put("SYNTHETIC_RUN", lambda: bool(j1.get("synthetic", False)), False)
    L.append(_jl("LOAD_WARNINGS", warns))
    return "".join(L)


def load_results(work) -> dict:
    """집계 결과를 읽어 {상수 이름: 값} 으로 돌려준다."""
    ns: dict = {}
    exec(build_text(work), {"float": float}, ns)
    return ns


# --------------------------------------------------------------------------- 체크포인트
# 이 분석의 기준 실행(2026-10-02, 실제 데이터) 값. 같은 원본·같은 코드·같은 시드면 그대로 나와야 한다.
EXPECTED = [
    ("P36 법인 수", lambda D: D["N_SAMPLE"]["P36"], 3372, 0),
    ("35개월 법인 수", lambda D: D["N_SAMPLE"]["P35만"], 1412, 0),
    ("판정 가능 기업 (P36)", lambda D: D["V2_COUNTS"]["P36"]["judgeable"], 2511, 0),
    ("최초 신호 기업 (P36)", lambda D: D["V2_COUNTS"]["P36"]["detected"], 488, 0),
    ("신호 기업 중 약화 건수", lambda D: D["MAIN"]["obs"], 120, 0),
    ("신호 기업 수 (결과 계산 가능)", lambda D: D["MAIN"]["n1"], 487, 0),
    ("신호 기업 약화 비율 (%)", lambda D: D["MAIN"]["p_det"], 24.6, 0.05),
    ("비교 기업 약화 비율 (%, 표준화)", lambda D: D["MAIN"]["p_ctrl"], 5.3, 0.05),
    ("RR (36개월 법인)", lambda D: D["MAIN"]["rr"], 4.69, 0.01),
    ("RR 95% 신뢰구간 하한", lambda D: D["MAIN"]["lo"], 3.75, 0.02),
    ("RR 95% 신뢰구간 상한", lambda D: D["MAIN"]["hi"], 5.95, 0.02),
    ("RR (35개월 법인만, 독립 검증)", lambda D: D["SAMPLE_RR"][1][2], 4.29, 0.01),
    ("RR (35개월 이상 전체)", lambda D: D["SAMPLE_RR"][2][2], 4.59, 0.01),
    ("민감도 20가지 중 최소 RR", lambda D: min(v[2] for v in D["VARIANTS"]), 2.75, 0.01),
    ("민감도 20가지 중 최대 RR", lambda D: max(v[2] for v in D["VARIANTS"]), 6.47, 0.01),
    ("신뢰구간 하한이 1 이하인 변형 수", lambda D: sum(1 for v in D["VARIANTS"] if v[3] <= 1), 0, 0),
    ("적중 비율 (%)", lambda D: D["HIT_P36"]["precision"], 24.6, 0.05),
    ("포착 비율 (%)", lambda D: D["HIT_P36"]["recall"], 10.8, 0.05),
    ("전년 같은 달 대비 신호의 RR", lambda D: D["YOY"]["P36_yoy"]["rr"], 4.36, 0.01),
]


def checkpoints(D: dict) -> pd.DataFrame:
    """이번 실행 값과 기준 실행 값의 비교표. 판정: 일치 / 차이."""
    rows = []
    for name, getter, expect, tol in EXPECTED:
        try:
            actual = getter(D)
        except (KeyError, IndexError, TypeError, ValueError):
            actual = None
        ok = actual is not None and abs(float(actual) - float(expect)) <= tol + 1e-9
        rows.append({"점검": name, "기준 실행(2026-10-02)": expect, "이번 실행": actual if actual is None else round(float(actual), 2), "판정": "일치" if ok else "차이"})
    return pd.DataFrame(rows)
