"""report_cnt(보고서 노트북의 실행기·결과 불러오기) 단위 점검 (원본·합성 데이터 불필요).  python test_report_cnt.py

노트북을 실제로 끝까지 돌리는 점검은 `python tools/smoke_test.py` (합성 데이터로 01~08 과 보고/종합분석보고서.ipynb 를 실행)가 맡는다.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import report_cnt as rc  # noqa: E402

ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# --- 노트북 -> 스크립트
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    nb = {"cells": [{"cell_type": "markdown", "source": ["# 제목\n"]},
                    {"cell_type": "code", "source": ["import os\n", "HOLDOUT_OPEN = False\n", "print(1)"]},
                    {"cell_type": "code", "source": ["x = 2"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = td / "t.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    s = rc.notebook_script(p)
    check("notebook_script: 코드 셀만 이어 붙임(마크다운 제외)", "제목" not in s and "print(1)" in s and "x = 2" in s)
    check("notebook_script: HOLDOUT_OPEN 을 환경변수로 연다(False 로 저장돼 있어도)", 'HOLDOUT_OPEN = os.environ.get("CNT_HOLDOUT_OPEN") == "1"' in s and "= False" not in s)
    nb["cells"][1]["source"] = ["HOLDOUT_OPEN = True\n"]
    p.write_text(json.dumps(nb), encoding="utf-8")
    check("notebook_script: True 로 저장돼 있어도 같은 방식", 'os.environ.get("CNT_HOLDOUT_OPEN")' in rc.notebook_script(p))

    # --- 끝난 단계 판정
    work = td / "work"
    (work / "interim").mkdir(parents=True)
    st = {s_[0]: s_ for s_ in rc.STEPS}
    check("step_done: 결과 파일이 없으면 끝나지 않음", not rc.step_done(work, st["01"]))
    (work / "interim" / "nb01_summary.json").write_text("{}", encoding="utf-8")
    check("step_done: 결과 파일이 있으면 끝남", rc.step_done(work, st["01"]))
    (work / "interim" / "nb05_summary.json").write_text(json.dumps({"holdout_open": False}), encoding="utf-8")
    check("step_done: 05 는 35개월 법인 잠금 상태(holdout_open=False)이면 끝나지 않음", not rc.step_done(work, st["05"]))
    (work / "interim" / "nb05_summary.json").write_text(json.dumps({"holdout_open": True}), encoding="utf-8")
    check("step_done: 05 는 잠금이 열린 요약이어야 끝남", rc.step_done(work, st["05"]))
    (work / "interim" / "nb05_summary.json").write_text("not json", encoding="utf-8")
    check("step_done: 05 요약이 깨져 있으면 끝나지 않음(오류 없이)", not rc.step_done(work, st["05"]))

    # --- 건너뛰기 (노트북을 실행하지 않고 반환값만 확인)
    for k, st_ in st.items():
        (work / "interim" / st_[2]).write_text(json.dumps({"holdout_open": True}), encoding="utf-8")
    res = rc.run_pipeline("없는경로.xlsx", work, force=False, verbose=False)
    check("run_pipeline: 끝난 8단계는 모두 건너뜀 (원본을 열지 않음)", [r["status"] for r in res] == ["skipped"] * 8)
    res = rc.run_pipeline("없는경로.xlsx", work, force=False, steps=["03", "04"], verbose=False)
    check("run_pipeline: steps 로 일부 단계만 지정 가능", [r["step"] for r in res] == ["03", "04"])
    check("STEPS 는 01~08 (보고서 노트북용), ALL_STEPS 는 09(P10 추가 민감도) 포함", [s_[0] for s_ in rc.STEPS] == [f"0{i}" for i in range(1, 9)] and [s_[0] for s_ in rc.ALL_STEPS][-1] == "09")
    check("09 의 노트북 파일이 실제로 있음", all((rc.NB_DIR / s_[1]).exists() for s_ in rc.ALL_STEPS))
    (work / "interim" / "nb09_summary.json").write_text("{}", encoding="utf-8")
    res = rc.run_pipeline("없는경로.xlsx", work, force=False, steps=["09"], verbose=False)
    check("run_pipeline: steps=['09'] 로 추가 민감도 단계를 지정할 수 있음(끝났으면 건너뜀), 기본 호출에는 09 가 들어가지 않음",
          [(r["step"], r["status"]) for r in res] == [("09", "skipped")])
    try:
        rc.run_pipeline("없는경로.xlsx", work, force=True, steps=["01"], verbose=False, expect_real_counts=False)
        failed = False
    except RuntimeError as e:
        failed = "단계 01" in str(e) and "로그" in str(e)
    check("run_pipeline: 단계가 실패하면 멈추고 단계 번호와 로그 위치를 알려 줌(원본이 없을 때)", failed)

# --- 상수 직렬화
ns = {}
exec(rc._jl("A", {"x": float("nan"), "y": [1, None, True], "z": "NaN 은 문자열"}), {"float": float}, ns)
check("_jl: NaN·None·True 를 파이썬 값으로 읽을 수 있게 직렬화 (문자열 안의 단어는 그대로)",
      ns["A"]["x"] != ns["A"]["x"] and ns["A"]["y"] == [1, None, True] and ns["A"]["z"] == "NaN 은 문자열")
check("_ci_text: '4.69 (3.75~5.95)' 해석, 해석 불가는 None", rc._ci_text("4.69 (3.75~5.95)") == (4.69, 3.75, 5.95) and rc._ci_text("기술만") is None)
check("_num: 쉼표 숫자와 가려진 칸('<5')", rc._num("1,412") == 1412.0 and rc._num("<5") is None and rc._num("-") is None)
check("_range: '1.2 ~ 3.4'", rc._range("1.2 ~ 3.4") == [1.2, 3.4])

# --- 체크포인트
D = {"N_SAMPLE": {"P36": 3372, "P35만": 1412}, "V2_COUNTS": {"P36": {"judgeable": 2511, "detected": 488}},
     "MAIN": {"obs": 120, "n1": 487, "p_det": 24.6, "p_ctrl": 5.3, "rr": 4.69, "lo": 3.75, "hi": 5.95},
     "SAMPLE_RR": [["a", 1, 4.69], ["b", 2, 4.29], ["c", 3, 4.59]], "VARIANTS": [["g", "l", 2.75, 2.2, 3.3], ["g", "l", 6.47, 4.7, 8.9]],
     "HIT_P36": {"precision": 24.6, "recall": 10.8}, "YOY": {"P36_yoy": {"rr": 4.36}}}
t = rc.checkpoints(D)
check("checkpoints: 기준 실행과 같은 값이면 모두 '일치'", len(t) == len(rc.EXPECTED) and (t["판정"] == "일치").all())
D2 = json.loads(json.dumps(D))
D2["MAIN"]["rr"] = 5.5
t2 = rc.checkpoints(D2)
check("checkpoints: 값이 다르면 그 행만 '차이'", list(t2[t2["판정"] == "차이"]["점검"]) == ["RR (36개월 법인)"])
t3 = rc.checkpoints({})
check("checkpoints: 값이 없어도 오류 없이 '차이'로 표시", (t3["판정"] == "차이").all())

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
