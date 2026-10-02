"""run_all.py(실행 진입점)와 strip_outputs.py(커밋 전 정리)의 단위 점검 (원본·합성 데이터 불필요).  python test_run_all.py

- run_all.py: 경로가 없거나 자리표시자이거나 파일이 아니거나 레포 폴더 안이면 계산을 시작하지 않고 안내만 낸다 / 알 수 없는 단계 거부
- strip_outputs.py: 임시 git 저장소에서 --check · 직접 정리 · --stage(작업 파일은 그대로, 인덱스에만 깨끗한 내용) 동작 확인
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def run(args, cwd=None, env=None):
    return subprocess.run([sys.executable, *args], cwd=str(cwd or ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


# --- run_all.py 입력 점검 (계산 전에 멈춤)
import os  # noqa: E402

clean_env = {k: v for k, v in os.environ.items() if k != "CNT_DATA_PATH"}
r = run(["run_all.py"], env=clean_env)
check("경로 없음 -> 종료코드 2 + 사용법 안내", r.returncode == 2 and "--data" in r.stderr)
r = run(["run_all.py", "--data", "C:/여기에_원본_경로.xlsx"], env=clean_env)
check("자리표시자 경로 -> 종료코드 2", r.returncode == 2 and "원본 경로가 없습니다" in r.stderr)
r = run(["run_all.py", "--data", "존재하지_않는_파일.xlsx"], env=clean_env)
check("없는 파일 -> 종료코드 2", r.returncode == 2 and "파일이 없습니다" in r.stderr)
with tempfile.TemporaryDirectory() as td:
    r = run(["run_all.py", "--data", td], env=clean_env)
    check("폴더를 줘도 -> 종료코드 2 (파일이어야 함)", r.returncode == 2 and "파일이 없습니다" in r.stderr)
inrepo = Path(__file__).resolve()   # 레포 안에 있는 파일 (이 시험 파일 자신)
r = run(["run_all.py", "--data", str(inrepo)], env=clean_env)
check("레포 폴더 안의 파일 -> 거부(실수로 커밋될 수 있음)", r.returncode == 2 and "레포 폴더 안" in r.stderr)
with tempfile.TemporaryDirectory() as td:
    f = Path(td) / "x.xlsx"
    f.write_bytes(b"x")
    r = run(["run_all.py", "--data", str(f), "--steps", "99"], env=clean_env)
    check("알 수 없는 단계 -> 종료코드 2", r.returncode == 2 and "알 수 없는 단계" in r.stderr)
    r = run(["run_all.py", "--help"], env=clean_env)
    check("--help 가 사용법을 보여 줌", r.returncode == 0 and "--data" in r.stdout and "--steps" in r.stdout)

# --- strip_outputs.py (임시 git 저장소)
NB = {"cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# 제목\n"]},
    {"cell_type": "code", "metadata": {"execution": {"iopub.execute_input": "2026-10-02T00:00:00Z"}}, "execution_count": 3,
     "outputs": [{"output_type": "stream", "name": "stdout", "text": ["집계값 123\n"]}],
     "source": ['DATA_PATH = os.environ.get("CNT_DATA_PATH", r"C:\\내폴더\\원본.xlsx")\n', "x = 1\n"]},
], "metadata": {"widgets": {"a": 1}, "kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")


with tempfile.TemporaryDirectory() as td:
    top = Path(td) / "repo"
    (top / "건수분석" / "tools").mkdir(parents=True)
    (top / "건수분석" / "notebooks").mkdir()
    (top / "건수분석" / "보고").mkdir()
    shutil.copy(ROOT / "tools" / "strip_outputs.py", top / "건수분석" / "tools" / "strip_outputs.py")
    for rel in ("notebooks/01_가.ipynb", "보고/종합.ipynb"):
        (top / "건수분석" / rel).write_text(json.dumps(NB, ensure_ascii=False), encoding="utf-8")
    git("init", "-q", cwd=top)
    tool = str(top / "건수분석" / "tools" / "strip_outputs.py")
    r = run([tool, "--check"])
    check("--check: 출력·로컬 경로가 있으면 종료코드 1, notebooks/ 와 보고/ 둘 다 검사", r.returncode == 1 and "01_가.ipynb" in r.stdout and "종합.ipynb" in r.stdout)
    before = (top / "건수분석" / "notebooks" / "01_가.ipynb").read_text(encoding="utf-8")
    r = run([tool, "--stage"])
    check("--stage: 성공", r.returncode == 0 and "2개" in r.stdout)
    after = (top / "건수분석" / "notebooks" / "01_가.ipynb").read_text(encoding="utf-8")
    check("--stage: 작업 파일은 그대로 (실행 결과 보존)", before == after and "집계값 123" in after)
    for rel in ("건수분석/notebooks/01_가.ipynb", "건수분석/보고/종합.ipynb"):
        blob = git("show", f":{rel}", cwd=top)
        d = json.loads(blob.stdout) if blob.returncode == 0 else {}
        code = [c for c in d.get("cells", []) if c["cell_type"] == "code"]
        src = "".join(code[0]["source"]) if code else ""
        check(f"--stage: 인덱스의 {rel} 는 출력·실행번호·실행시각·로컬 경로가 없음",
              blob.returncode == 0 and code and code[0]["outputs"] == [] and code[0]["execution_count"] is None
              and "execution" not in code[0]["metadata"] and "내폴더" not in src and "여기에_원본_경로.xlsx" in src and "x = 1" in src
              and "widgets" not in d["metadata"] and d["metadata"].get("kernelspec") == {"name": "python3"})
    r = run([tool])
    check("직접 정리: 작업 파일에서 출력 제거", r.returncode == 0 and "집계값 123" not in (top / "건수분석" / "notebooks" / "01_가.ipynb").read_text(encoding="utf-8"))
    r = run([tool, "--check"])
    check("정리 후 --check 는 종료코드 0", r.returncode == 0)

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
