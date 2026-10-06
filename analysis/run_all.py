"""원본 파일 경로만 넣으면 건수분석 전체(노트북 01~09의 계산)를 같은 코드·같은 난수 시드로 실행한다. Jupyter 불필요.

    python run_all.py --data "D:\\원본폴더\\법인데이터.xlsx"
    python run_all.py --data "..." --boot 300      # 부트스트랩 횟수를 줄인 빠른 시험 실행 (기준 실행과 값이 조금 달라짐)
    python run_all.py --data "..." --steps 05 06   # 일부 단계만
    python run_all.py --data "..." --force         # 끝난 단계도 처음부터 다시 계산
    python run_all.py --data "..." --skip-extra    # 09(P10 추가 민감도) 제외

원본 경로는 환경변수 CNT_DATA_PATH 로 줘도 된다. 원본은 레포 밖에 둔다 (레포 안이면 실수로 커밋될 수 있어 거부함).
산출물(집계 CSV·JSON, 그림, 행 단위 npz)은 레포의 data/work_cnt/ 에 저장된다 (.gitignore 대상 -- 커밋·공유하지 않는다).
단계마다 이미 끝난 결과가 있으면 건너뛴다. 끝나면 기준 실행(2026-10-02) 값과 비교한 점검표를 보여 준다.
화면에는 단계 이름·시간·점검표(집계값)만 나오고, 상세 출력은 data/work_cnt/logs/ 에 저장된다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))

PLACEHOLDER = "여기에_원본_경로"
CORE_STEPS = ["01", "02", "03", "04", "05", "06", "07", "08"]
EXTRA = ["09"]


def _fail(msg: str, code: int = 2) -> "None":
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="원본 파일 경로만 넣어 건수분석 01~09 를 실행한다.")
    ap.add_argument("--data", help="원본 파일 경로(.xlsx/.csv). 생략하면 환경변수 CNT_DATA_PATH")
    ap.add_argument("--sheet", default=os.environ.get("CNT_SHEET", "all"), help='엑셀 시트: "all"(전부, 기본) 또는 번호/이름')
    ap.add_argument("--work", default=None, help="산출물 폴더 (기본: 레포의 data/work_cnt)")
    ap.add_argument("--boot", type=int, default=None, help="부트스트랩 횟수 (기본: 노트북 값 = 기준 실행과 같음)")
    ap.add_argument("--steps", nargs="+", default=None, help="실행할 단계 번호 (예: 05 06). 기본: 01~09")
    ap.add_argument("--force", action="store_true", help="끝난 단계도 처음부터 다시 계산")
    ap.add_argument("--skip-extra", action="store_true", help="09(P10 추가 민감도)를 건너뜀")
    ap.add_argument("--no-expect", action="store_true", help="01 의 행·법인 수 기대값 점검을 끔 (합성·다른 버전의 데이터일 때)")
    ap.add_argument("--no-check", action="store_true", help="끝난 뒤 기준 실행 점검표를 만들지 않음")
    a = ap.parse_args(argv)

    missing = []
    for mod in ("numpy", "pandas", "matplotlib", "openpyxl"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        _fail(f"필요한 패키지가 없습니다: {', '.join(missing)}\n  pip install -r requirements.txt  (이 폴더에 있음)")

    import cntlib as cl
    import report_cnt as rc

    data = a.data or os.environ.get("CNT_DATA_PATH")
    if not data or PLACEHOLDER in data:
        _fail('원본 경로가 없습니다.\n  python run_all.py --data "원본파일경로.xlsx"\n  (또는 환경변수 CNT_DATA_PATH 에 경로를 넣으세요)')
    dp = Path(data)
    if not dp.is_file():
        _fail("원본 경로에 파일이 없습니다. 경로를 확인하세요 (따옴표로 감싸고, 폴더가 아니라 파일을 가리켜야 합니다).")
    try:
        rel = dp.resolve().relative_to(cl.REPO_ROOT)
    except ValueError:
        rel = None
    if rel is not None and rel.parts[:1] != ("data",):
        _fail("원본이 레포 폴더 안에 있습니다. 실수로 커밋될 수 있으므로 레포 밖(또는 레포의 data/ 아래)으로 옮기세요.")

    steps = a.steps or (CORE_STEPS if a.skip_extra else CORE_STEPS + EXTRA)
    known = {s[0] for s in rc.ALL_STEPS}
    bad = [s for s in steps if s not in known]
    if bad:
        _fail(f"알 수 없는 단계: {', '.join(bad)} (가능: {', '.join(sorted(known))})")

    work = cl.resolve_work_dir(a.work)
    print(f"산출물 폴더: {work}")
    print(f"원본 파일: (설정됨) | 단계: {' '.join(steps)} | 처음부터 다시 계산: {a.force}")
    t0 = time.time()
    try:
        res = rc.run_pipeline(data, work, sheet=a.sheet, boot=a.boot, force=a.force, steps=steps, expect_real_counts=not a.no_expect)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        return 1
    done = sum(r["status"] == "done" for r in res)
    print(f"\n끝: 실행 {done}단계 · 건너뜀 {len(res) - done}단계 · {time.time() - t0:.0f}초   (상세 로그: {work / 'logs'})")

    if not a.no_check and all(s in steps for s in CORE_STEPS):
        try:
            D = rc.load_results(work)
            if D.get("SYNTHETIC_RUN"):
                print(cl.SYNTHETIC_NOTICE)
            chk = rc.checkpoints(D)
            print("\n[기준 실행(2026-10-02)과의 점검표]")
            print(chk.to_string(index=False))
            n_diff = int((chk["판정"] == "차이").sum())
            if n_diff == 0:
                print("\n모든 점검 값이 기준 실행과 일치함 -> 같은 결과를 재현했음.")
            else:
                print(f"\n{n_diff}개 값이 기준 실행과 다름. 합성 데이터이거나 다른 원본·설정(--boot 등)이면 정상.")
                print("같은 원본·기본 설정인데 다르면 01 의 검증표(logs/01.log)와 설정을 확인하세요.")
        except Exception as e:  # 점검표가 안 나와도 계산 결과는 이미 저장되어 있음
            print(f"\n점검표를 만들지 못했습니다 ({type(e).__name__}). 계산 결과는 {work} 에 저장되어 있습니다.")
    if "09" in steps:
        print(f"09 결과: {work / 'interim' / 'nb09_summary.json'} 와 {work / 'output' / 'nb09'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
