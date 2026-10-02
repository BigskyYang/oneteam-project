"""노트북의 실행 결과(출력)와 로컬 경로를 지운다 -- 커밋하기 전에 반드시 실행.

노트북을 실행하면
  1) 출력 셀에 데이터에서 나온 집계값과 그림이 저장되고,
  2) 설정 셀의 DATA_PATH 에 내 컴퓨터의 원본 파일 경로가 들어간다.
둘 다 레포에 올라가면 안 되므로, 커밋 전에 출력을 지우고 DATA_PATH 를 자리표시자로 되돌린다.
대상: notebooks/ 와 보고/ 안의 모든 .ipynb

    python strip_outputs.py            # 작업 파일을 직접 정리 (실행 결과가 사라지므로 필요하면 먼저 복사해 둘 것)
    python strip_outputs.py --check    # 정리할 것이 남아 있으면 목록을 보여 주고 종료코드 1 (바꾸지 않음)
    python strip_outputs.py --stage    # 작업 파일은 그대로 두고, 정리한 내용만 git 인덱스(커밋 대기 목록)에 올림
                                       #  -> 내 컴퓨터의 실행 결과는 남기고, 커밋에는 깨끗한 노트북만 들어감
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIRS = [ROOT / "notebooks", ROOT / "보고"]
PLACEHOLDER = "여기에_원본_경로.xlsx"
PATH_PAT = re.compile(r'(os\.environ\.get\("CNT_DATA_PATH", r")([^"\n]*)("\))')


def notebooks() -> list[Path]:
    return [f for d in NB_DIRS if d.exists() for f in sorted(d.glob("*.ipynb"))]


def has_outputs(nb: dict) -> bool:
    return any(c.get("cell_type") == "code" and (c.get("outputs") or c.get("execution_count") is not None) for c in nb.get("cells", []))


def has_local_path(nb: dict) -> bool:
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            m = PATH_PAT.search("".join(c.get("source", [])))
            if m and m.group(2) != PLACEHOLDER:
                return True
    return False


def strip(nb: dict) -> dict:
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
            c.get("metadata", {}).pop("execution", None)   # 실행 시각 기록
            src = "".join(c.get("source", []))
            new = PATH_PAT.sub(lambda m: m.group(1) + PLACEHOLDER + m.group(3), src)
            if new != src:
                c["source"] = new.splitlines(keepends=True)
    nb.get("metadata", {}).pop("widgets", None)
    return nb


def dump(nb: dict) -> str:
    return json.dumps(nb, ensure_ascii=False, indent=1) + "\n"


def _git(args: list[str], cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} 실패: {r.stderr.strip()}")
    return r.stdout.strip()


def stage(f: Path, nb: dict) -> None:
    """정리한 내용을 git 인덱스에만 올린다 (작업 파일은 건드리지 않음)."""
    top = Path(_git(["rev-parse", "--show-toplevel"], f.parent)).resolve()
    rel = f.resolve().relative_to(top).as_posix()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "clean.ipynb"
        tmp.write_text(dump(strip(nb)), encoding="utf-8", newline="\n")
        sha = _git(["hash-object", "-w", "--path", rel, str(tmp)], top)
    _git(["update-index", "--add", "--cacheinfo", f"100644,{sha},{rel}"], top)


def main() -> int:
    check = "--check" in sys.argv
    do_stage = "--stage" in sys.argv
    dirty, staged = [], []
    for f in notebooks():
        nb = json.loads(f.read_text(encoding="utf-8"))
        why = [w for w, bad in (("출력", has_outputs(nb)), ("로컬 경로", has_local_path(nb))) if bad]
        label = f"{f.parent.name}/{f.name}"
        if why:
            dirty.append(f"{label} ({', '.join(why)})")
        if do_stage and not check:
            stage(f, nb)
            staged.append(label)
        elif why and not check:
            f.write_text(dump(strip(nb)), encoding="utf-8", newline="\n")
    if check:
        if dirty:
            print("정리가 필요한 노트북:", "; ".join(dirty))
            print("python tools/strip_outputs.py 로 정리하거나, 작업 파일을 두고 python tools/strip_outputs.py --stage 로 커밋 대기 목록에만 올리세요.")
            return 1
        print("모든 노트북에 출력과 로컬 경로가 없습니다.")
        return 0
    if do_stage:
        print(f"깨끗한 내용을 커밋 대기 목록에 올렸습니다 ({len(staged)}개). 작업 파일은 그대로입니다:", "; ".join(staged))
        return 0
    print("정리했습니다:", "; ".join(dirty) if dirty else "(정리할 것이 없음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
