"""노트북 점검 -- 합성(가짜) 데이터로 노트북 코드를 처음부터 끝까지 돌려 본다. 실제 데이터는 쓰지 않는다.

    python smoke_test.py

확인하는 것
  1. 각 노트북의 코드 셀이 오류 없이 끝까지 실행된다 (01 -> 02 -> ... -> 08 순서, 이어서 09 추가 민감도, 보고/종합분석보고서.ipynb, 실행 진입점 run_all.py)
  2. 합성 데이터에 심어 둔 구조를 분석 코드가 찾아낸다 (표본 크기, 인터넷↔스마트 이동, 외환의 독립성)
  3. 저장된 표·요약에 법인ID(SYN_...)가 없다
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "code"))
from make_synthetic_cnt import make  # noqa: E402

ok_all = True


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not cond else ""))


def code_of(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    # 05: 사용자가 잠금을 해제해 저장했어도 점검은 잠금·해제 두 분기를 환경변수로 따로 돌린다
    return text.replace("HOLDOUT_OPEN = True\n", 'HOLDOUT_OPEN = os.environ.get("CNT_HOLDOUT_OPEN") == "1"\n')


def main() -> int:
    nbs = sorted((ROOT / "notebooks").glob("0[1-8]*.ipynb"))
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        df, exp = make(1500, seed=3)
        csv = td / "syn.csv"
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"합성 데이터: 행 {exp['rows']:,}, 법인 {exp['firms']:,}, P36 {exp['p36']:,}, P35 {exp['p35']:,}, "
              f"이동형 {exp['migrators']}, 위축형 {exp['shrinkers']}")
        env = {**os.environ, "CNT_DATA_PATH": str(csv), "CNT_NO_EXPECT": "1", "CNT_WORK_DIR": str(td / "work"), "CNT_BOOT": "1000",
               "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
        for nb in nbs:
            script = td / (nb.stem + ".py")
            script.write_text(code_of(nb), encoding="utf-8")
            r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT / "notebooks"), env=env, capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            check(f"{nb.name} 끝까지 실행", r.returncode == 0, (r.stdout[-1500:] + r.stderr[-1500:]))
            if r.returncode == 0:
                check(f"{nb.name} 합성 경고 출력", "합성(가짜)" in r.stdout)
                check(f"{nb.name} 출력에 법인ID 없음", "SYN_" not in r.stdout + r.stderr)

        # 05(입금 검증): 잠금 상태 요약을 먼저 읽고, 같은 노트북을 잠금 해제로 한 번 더 실행해 35개월 법인 분기를 점검한다
        wk0 = td / "work" / "interim"
        s5_locked = json.loads((wk0 / "nb05_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb05_summary.json").exists() else {}
        nb5s = [x for x in nbs if x.name.startswith("05")]
        s5_open = {}
        if nb5s:
            script5 = td / "05_open.py"
            script5.write_text(code_of(nb5s[0]), encoding="utf-8")
            r5 = subprocess.run([sys.executable, str(script5)], cwd=str(ROOT / "notebooks"), env={**env, "CNT_HOLDOUT_OPEN": "1", "CNT_BOOT": "300"}, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
            check("05 (잠금 해제) 끝까지 실행", r5.returncode == 0, (r5.stdout[-1200:] + r5.stderr[-1200:]))
            s5_open = json.loads((wk0 / "nb05_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb05_summary.json").exists() else {}
        wk = td / "work" / "interim"
        s1 = json.loads((wk / "nb01_summary.json").read_text(encoding="utf-8")) if (wk / "nb01_summary.json").exists() else {}
        check("01: P36·P35 크기가 정답과 일치", s1.get("P36") == exp["p36"] and s1.get("P35") == exp["p35"], str(s1))
        check("01: P35+ = P36 + P35", s1.get("P35p") == exp["p36"] + exp["p35"])
        check("01: 빈 달 위치 비율이 심은 값(20/60/20%) 근처",
              abs(s1.get("missing_first", 0) / max(exp["p35"], 1) - 0.2) < 0.12 and abs(s1.get("missing_mid", 0) / max(exp["p35"], 1) - 0.6) < 0.15,
              str(s1))
        s2 = json.loads((wk / "quant_spec.json").read_text(encoding="utf-8")) if (wk / "quant_spec.json").exists() else {}
        check("02: 수치화 결정 기록 저장", "aggregate_activity" in s2 and s2.get("cap_primary", s2.get("approx_count", {}).get("cap_primary")) == 60.0, str(s2)[:200])
        cs = wk.parent / "output" / "nb02" / "i1_cap_sensitivity.csv"
        if cs.exists():
            import pandas as pd
            t = pd.read_csv(cs, encoding="utf-8-sig")
            vals = pd.to_numeric(t[t["지표"].str.contains(r"cap100", regex=False)]["값"], errors="coerce")
            # cap 51 -> 100 에서 I1 이 늘 수 있는 최대 비율은 100/51 - 1 = 96.1% (분모 0 처리 오류 방지 점검)
            check("02: cap 민감도 지표가 이론 최대(96.1%)를 넘지 않음", len(vals) >= 4 and float(vals.max()) <= 96.2, str(vals.tolist()))
        s3 = json.loads((wk / "nb03_summary.json").read_text(encoding="utf-8")) if (wk / "nb03_summary.json").exists() else {}
        check("03: 심어 둔 이동(인터넷→스마트)을 찾음: 기업 내 상관이 음수", s3.get("within_corr_internet_smart_P36", 1) < -0.03, str(s3))
        check("03: 반대 방향 lift(인터넷↓·스마트↑) > 1.2 이고 같은 방향 lift 보다 큼",
              s3.get("lift_opp_internet_down_smart_up_P36", 0) > 1.2
              and s3.get("lift_opp_internet_down_smart_up_P36", 0) > s3.get("lift_same_internet_down_smart_down_P36", 9), str(s3))
        check("03: 총 활동 유지 창만 봐도 반대 방향 lift > 1.2 (이동은 총 활동 유지 조건에서도 보임)",
              s3.get("lift_opp_stable_internet_down_smart_up_P36", 0) > 1.2, str(s3))
        check("03: 확증 1쌍(인터넷↓→스마트↑)이 전체 창 CI 하한 > 1 이고 같은 방향보다 커서 지지됨", s3.get("confirmatory_supported_P36") is True, str(s3))
        check("03: 외환은 국내 채널과 독립 (기업 내 |상관| < 0.1)", s3.get("fx_max_abs_within_corr_with_domestic", 1) < 0.10, str(s3))
        check("03: P36 과 P35+ 결과가 비슷 (기업 내 상관 최대 차이 < 0.1)", s3.get("pop_max_within_corr_diff", 1) < 0.10, str(s3))
        s4 = json.loads((wk / "detect_spec.json").read_text(encoding="utf-8")) if (wk / "detect_spec.json").exists() else {}
        rule = s4.get("rule", {})
        check("04: 탐지 규칙 저장 (n 은 양수, k 는 후보 중 하나)", rule.get("n_pct", 0) >= 5 and rule.get("k") in (1, 2, 3, 6), str(rule)[:200])
        check("04: P36 에서 탐지 기업이 존재", s4.get("power", {}).get("P36", {}).get("N1", 0) > 0, str(s4.get("power"))[:200])
        check("04: 행 단위 결과 파일 저장", (wk / "detect_result.npz").exists())
        s4b = json.loads((wk / "detect_spec_v2.json").read_text(encoding="utf-8")) if (wk / "detect_spec_v2.json").exists() else {}
        check("04: 수정 규칙(v2) 저장 (천장 칸 제외 하한, n 양수)", s4b.get("rule", {}).get("floor_excludes_ceiling_cells") is True and s4b.get("rule", {}).get("n_pct", 0) >= 5, str(s4b)[:200])
        check("05: 잠금 상태에서는 35개월 법인의 입금 결과를 계산하지 않음", s5_locked.get("holdout_open") is False and "holdout" not in s5_locked, str(list(s5_locked.keys())))
        m5 = s5_locked.get("P36", {}).get("main", {})
        check("05: P36 H-C1 결과 저장 (RR 정의, 심어 둔 입금 약화를 탐지 RR > 1.3)", (m5.get("rr") or 0) > 1.3, str(m5)[:200])
        check("05: 경로 해석 문구에 '선행'·'조기경보' 표현이 없음", "선행" not in s5_locked.get("P36", {}).get("path", {}).get("interpretation", "선행") and
              "조기경보" not in s5_locked.get("P36", {}).get("path", {}).get("interpretation", "조기경보"))
        check("05: 잠금 해제 시 P35만·P35+ 결과 저장", set(s5_open.get("holdout", {}).keys()) == {"P35만", "P35+"} and s5_open.get("holdout_open") is True, str(list(s5_open.keys())))
        ex5 = json.loads((wk0 / "nb05_explore.json").read_text(encoding="utf-8")) if (wk0 / "nb05_explore.json").exists() else {}
        check("05: 사후 탐색(9-B) 결과 저장 (급증 꼬리·급등 여부별 집단, RR 정의)", {"up_tail", "surge_groups"} <= set(ex5.keys()) and (ex5.get("up_tail", {}).get("rr") is not None)
              and (ex5.get("surge_groups", {}).get("전체", {}).get("rr") is not None), str(ex5)[:300])
        s6 = json.loads((wk0 / "nb06_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb06_summary.json").exists() else {}
        check("06: 유형 경계·자동이체 경계 저장, 유형이 2개 이상, P36·P35만 C1~C4 요약 저장",
              len(s6.get("type_edges", [])) == 2 and len(s6.get("present_types", [])) >= 2 and {"c1", "c2", "c3", "c4_kept"} <= set(s6.get("P36", {}).keys())
              and {"c1", "c2", "c3", "c4_kept"} <= set(s6.get("P35", {}).keys()), str(s6)[:300])
        c2s = [v for v in s6.get("P36", {}).get("c2", {}).values() if v]
        check("06: 유형별 배수(RR)가 정의되고 심어 둔 입금 약화로 1보다 큼 (합성)", len(c2s) >= 2 and all((v.get("rr") or 0) > 1 for v in c2s), str(c2s)[:300])
        s7 = json.loads((wk0 / "nb07_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb07_summary.json").exists() else {}
        v7 = s7.get("variants", {})
        check("07: 기준과 변형 15개 이상의 RR 저장, 선택 절차 재실행·문턱 일치도 저장", (s7.get("base", {}).get("rr") or 0) > 1.3 and len(v7) >= 15
              and {"s30", "s35", "same_rule"} <= set(s7.get("selection", {}).keys()) and s7.get("threshold_overlap", {}).get("jaccard") is not None, str(s7)[:300])
        cap7 = [v for k, v in v7.items() if "cap 51" in k or "cap 100" in k]
        check("07: 같은 표본 변형에는 기준 대비 비가 짝지어 계산되고 (cap 변형) 합성에서 연관이 유지됨(RR > 1)", len(cap7) == 2 and all(v.get("ratio") and (v.get("rr") or 0) > 1 for v in cap7), str(cap7)[:300])
        s8 = json.loads((wk0 / "nb08_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb08_summary.json").exists() else {}
        a8, b8 = s8.get("A", {}), s8.get("B", {})
        t36 = a8.get("two_by_two", {}).get("P36", {})
        check("08: 2×2 표(P36·P35)·기업 단위 요약·기준 n 격자 9개 저장, 합성에서 적중 비율 > 비신호 약화율",
              {"P36", "P35"} <= set(a8.get("two_by_two", {}).keys()) and len(a8.get("grid", [])) == 9 and {"early", "late", "none"} <= set(a8.get("firm_timing", {}).get("P36", {}).keys())
              and (t36.get("precision") or 0) > (t36.get("ctrl_rate") or 1), str(a8)[:300])
        k8 = int(s4b.get("rule", {}).get("k", 0))          # 합성 데이터에서는 k 가 6 이 아닐 수 있다 (실제 데이터는 6)
        want_prior = len(set(range(20 - k8 + 1 - k8, 20 + 1 - k8)) & set(range(9, 15)))
        check("08: 신호의 비교 달과 약화의 기준 달이 겹치는 달 수: 직전 기간 규칙은 k 로 계산한 값(k=6 이면 6), 전년 규칙은 0",
              k8 in (1, 2, 3, 6) and b8.get("overlap", {}).get("prior", {}).get("overlap_months") == want_prior and b8.get("overlap", {}).get("yoy", {}).get("overlap_months") == 0
              and (k8 != 6 or want_prior == 6), f"k={k8} want={want_prior} {b8.get('overlap')}")
        check("08: 전년 규칙의 RR·기준 대비 비 저장 (합성에서 연관 유지: RR > 1)", (b8.get("P36_yoy", {}).get("rr") or 0) > 1.0 and b8.get("P36_yoy", {}).get("ratio") is not None
              and (b8.get("yoy_rule", {}).get("n") or 0) >= 5, str(b8)[:300])
        # 09(추가 민감도: 후보월 확장·대칭 결과 변수): 01~04 가 끝난 작업 폴더에서 돌린다 (05 이후 결과는 필요 없다)
        nb9x = sorted((ROOT / "notebooks").glob("09*.ipynb"))
        if nb9x:
            script9x = td / "09x.py"
            script9x.write_text(code_of(nb9x[0]), encoding="utf-8")
            r9x = subprocess.run([sys.executable, str(script9x)], cwd=str(ROOT / "notebooks"), env=env, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace")
            check(f"{nb9x[0].name} 끝까지 실행", r9x.returncode == 0, (r9x.stdout[-1500:] + r9x.stderr[-1500:]))
            if r9x.returncode == 0:
                check("09 합성 경고 출력", "합성(가짜)" in r9x.stdout)
                check("09 출력에 법인ID 없음", "SYN_" not in r9x.stdout + r9x.stderr)
                s9 = json.loads((wk0 / "nb09_summary.json").read_text(encoding="utf-8")) if (wk0 / "nb09_summary.json").exists() else {}
                sc = s9.get("scenarios", [])
                check("09: 시나리오 10개(A·B1·B2·C1·C2 x 36개월·35개월만) 저장, 확장 첫 후보월 = k-1+L 이고 본 분석 첫 후보월(k+11) 이하(직전 비교면 11, 전년 비교면 같음)",
                      len(sc) == 10 and {x["sample"] for x in sc} == {"36개월 법인", "35개월 법인만"}
                      and s9.get("rule", {}).get("e_lo_ext") == s9["rule"]["k"] - 1 + (6 if s9["rule"]["mode"] == "직전" else 12) and s9["rule"]["e_lo_ext"] <= s9["rule"]["e_lo_base"], str(s9)[:300])
                check("09: 기준 RR 이 07 의 기준 RR 과 같음 (같은 규칙·시드)", abs((s9.get("base", {}).get("P36", {}).get("rr") or 0) - (s7.get("base", {}).get("rr") or 0)) < 1e-9,
                      f"{s9.get('base', {}).get('P36', {}).get('rr')} vs {s7.get('base', {}).get('rr')}")
                b1 = [x for x in sc if x["group"].startswith("B") and "6개월" in x["label"] and x["sample"] == "36개월 법인"]
                check("09: 합성에서 대칭 결과(직전 6개월 대비)로도 연관 유지 (RR > 1)", len(b1) == 1 and (b1[0].get("rr") or 0) > 1.0, str(b1)[:300])
                a_ = [x for x in sc if x["group"].startswith("A") and x["sample"] == "36개월 법인"]
                check("09: 후보월 확장에서도 RR 이 계산되고 기준과 같은 기업을 포함 (탐지 기업 수 >= 기준)", len(a_) == 1 and (a_[0].get("rr") or 0) > 1.0 and a_[0]["n_det"] >= s9["base"]["P36"]["n_det"], str(a_)[:300])
        # 보고/종합분석보고서.ipynb (원본 실행 버전): 별도의 (조금 더 큰) 합성 데이터로 01~08 계산부터 보고서 1~11장 그림·표까지 한 번에 돌린다
        #   (1,500곳짜리 합성 데이터는 집단이 너무 작아 일부 표가 '기술만'으로 비므로 6,000곳을 쓴다)
        nb9 = sorted((ROOT / "보고").glob("종합분석보고서*.ipynb"))
        if nb9:
            df9, exp9 = make(6000, seed=3)
            csv9 = td / "syn9.csv"
            df9.to_csv(csv9, index=False, encoding="utf-8-sig")
            script9 = td / "report.py"
            script9.write_text(code_of(nb9[0]), encoding="utf-8")
            env9 = {**env, "CNT_DATA_PATH": str(csv9), "CNT_WORK_DIR": str(td / "work9"), "CNT_BOOT": "300"}
            r9 = subprocess.run([sys.executable, str(script9)], cwd=str(ROOT / "notebooks"), env=env9, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
            check(f"보고/{nb9[0].name} 끝까지 실행 (01~08 계산 + 보고서)", r9.returncode == 0, (r9.stdout[-1500:] + r9.stderr[-1500:]))
            if r9.returncode == 0:
                check("보고서 노트북 합성 경고 출력", "합성(가짜)" in r9.stdout)
                check("보고서 노트북 출력에 법인ID 없음", "SYN_" not in r9.stdout + r9.stderr)
                check("보고서 노트북: 01~08 여덟 단계를 모두 실행 (건너뛰지 않음)", r9.stdout.count("실행 중") == 8 and "건너뜀" not in r9.stdout)
                # 이미 끝난 작업 폴더를 다시 읽는 두 번째 실행은 건너뛴다
                r9b = subprocess.run([sys.executable, str(script9)], cwd=str(ROOT / "notebooks"), env=env9, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace")
                check("보고서 노트북 두 번째 실행: 끝난 8단계는 건너뛰고 보고서만 다시 그림", r9b.returncode == 0 and r9b.stdout.count("건너뜀") == 8, (r9b.stdout[-800:] + r9b.stderr[-800:]))
                # 팀원용 실행 진입점: 원본 경로만 주면 01~09 를 돌린다 (끝난 01~08 은 건너뛰고 09 를 실행, 마지막에 점검표)
                rcli = subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--data", str(csv9), "--work", str(td / "work9"), "--no-expect", "--boot", "300"],
                                      cwd=str(ROOT), env=env9, capture_output=True, text=True, encoding="utf-8", errors="replace")
                check("run_all.py: 끝난 01~08 은 건너뛰고 09 를 실행, 합성 경고와 기준 실행 점검표 출력",
                      rcli.returncode == 0 and rcli.stdout.count("] 건너뜀") == 8 and "[09] 실행 중" in rcli.stdout and "점검표" in rcli.stdout and "합성(가짜)" in rcli.stdout,
                      (rcli.stdout[-1200:] + rcli.stderr[-1200:]))
                check("run_all.py: 09 결과 요약 저장, 출력에 법인ID 없음", (td / "work9" / "interim" / "nb09_summary.json").exists() and "SYN_" not in rcli.stdout + rcli.stderr)
            leaked9 = [p.name for p in (td / "work9").rglob("*") if p.is_file() and p.suffix in (".csv", ".json", ".log")
                       and "SYN_" in p.read_text(encoding="utf-8-sig", errors="ignore")]
            check("보고서 노트북 작업 폴더의 표·요약·로그에 법인ID 없음", not leaked9, str(leaked9))
        leaked = [p.name for p in (td / "work").rglob("*") if p.is_file() and p.suffix in (".csv", ".json")
                  and "SYN_" in p.read_text(encoding="utf-8-sig", errors="ignore")]
        check("저장된 표·요약에 법인ID 없음", not leaked, str(leaked))
    print("\n결과:", "모두 통과" if ok_all else "실패 있음")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
