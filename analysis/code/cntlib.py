"""거래 건수 기반 분석 공용 도구 (건수분석 노트북에서 사용).

레포 README '데이터 취급 원칙'을 코드로 강제한다.
- 원본 경로는 노트북 설정 셀에서만 받고, 레포 안에 복사하지 않는다.
- 법인ID는 읽자마자 내부 정수 코드로 바꾸고 어디에도 남기지 않는다.
- 화면 출력은 집계값만, 5 미만 칸은 가린다(sup).
- 행 단위 중간 산출물(패널 등)은 레포의 data/ 아래(.gitignore 대상) 또는 레포 밖에만 저장한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # <root>/analysis/code/cntlib.py

KEY_ID, KEY_YM = "법인ID", "기준년월"
ATTR_COLS = ["업종_대분류", "사업장_시도", "법인_고객등급", "전담고객여부"]
AMT_COLS = ["요구불입금금액", "요구불출금금액", "요구불예금잔액"]  # 탐지에는 쓰지 않는다. 결과(검증)용
CNT_COLS = ["창구거래건수", "인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수", "ATM거래건수", "자동이체거래건수",
            "외환_수출실적거래건수", "외환_수입실적거래건수"]
CH5 = CNT_COLS[:5]  # 국내 채널 5개
AUTO = CNT_COLS[5]  # 자동이체
FX = CNT_COLS[6:]  # 외환 (수출, 수입)
SHORT = {"창구거래건수": "창구", "인터넷뱅킹거래건수": "인터넷", "스마트뱅킹거래건수": "스마트", "폰뱅킹거래건수": "폰",
         "ATM거래건수": "ATM", "자동이체거래건수": "자동이체", "외환_수출실적거래건수": "외환수출", "외환_수입실적거래건수": "외환수입"}
ONLINE = ["인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수"]
OFFLINE = ["창구거래건수", "ATM거래건수"]
CORE_COLS = [KEY_ID, KEY_YM] + ATTR_COLS + AMT_COLS + CNT_COLS

START_YM, END_YM, T = 202301, 202512, 36
N_BANDS = 10
MIN_CELL = 5
# 구간 코드 0..9 의 대표 건수. 9('50건초과')는 열린 구간이라 cap 으로 따로 정한다.
BAND_MIDS = [0.0, 1.0, 2.0, 4.0, 8.0, 15.5, 25.5, 35.5, 45.5]
DEFAULT_CAP = 60.0

SYNTHETIC_NOTICE = "※ 합성(가짜) 데이터로 만든 결과입니다. 코드 검증용이며 실제 결과가 아닙니다."


class CntError(Exception):
    """데이터가 기대와 다를 때 조용히 넘어가지 않고 멈추기 위한 예외. 메시지에는 데이터 값을 넣지 않는다."""


# --------------------------------------------------------------------------- 출력 보호
def sup(n, min_n: int = MIN_CELL) -> str:
    n = int(n)
    return f"<{min_n}" if 0 < n < min_n else f"{n:,}"


def pct(num, den, d: int = 1) -> str:
    return "-" if den == 0 else f"{num / den * 100:.{d}f}"


def resolve_work_dir(arg=None) -> Path:
    """산출물 폴더. 레포 안이면 data/ 아래만 허용한다(.gitignore 대상)."""
    d = Path(arg).resolve() if arg else (REPO_ROOT / "data" / "work_cnt")
    try:
        rel = d.relative_to(REPO_ROOT)
    except ValueError:
        rel = None
    if rel is not None and rel.parts[:1] != ("data",):
        raise SystemExit("산출물은 레포의 data/ 아래 또는 레포 밖에만 저장할 수 있습니다 (그 밖의 폴더는 실수로 커밋될 수 있음).")
    d.mkdir(parents=True, exist_ok=True)
    return d


def show_table(df: pd.DataFrame, title: str | None = None, save_to: Path | None = None, name: str | None = None) -> None:
    """노트북에서는 표로, 일반 파이썬에서는 텍스트로 보여 주고 CSV로도 저장한다."""
    if title:
        print(f"\n[{title}]")
    try:
        from IPython.display import display

        display(df)
    except Exception:
        print(df.to_string(index=False))
    if save_to is not None and name:
        save_to.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to / f"{name}.csv", index=False, encoding="utf-8-sig")


def setup_korean_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]:
        if n in names:
            plt.rcParams["font.family"] = n
            break
    plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- 원본 읽기
def read_raw(path, sheet="0") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise CntError("DATA_PATH 경로에 파일이 없습니다. 설정 셀의 경로를 확인하세요.")
    if p.suffix.lower() == ".csv":
        try:
            return pd.read_csv(p, usecols=CORE_COLS, dtype=str, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(p, usecols=CORE_COLS, dtype=str, encoding="cp949")
    sh = None if str(sheet).lower() == "all" else (int(sheet) if str(sheet).isdigit() else sheet)
    df = pd.read_excel(p, sheet_name=sh, usecols=CORE_COLS, dtype=str, engine="openpyxl")
    if isinstance(df, dict):
        df = pd.concat(df.values(), ignore_index=True)
    return df


# --------------------------------------------------------------------------- 구간 라벨 -> 코드
def band_sort_key(label) -> float:
    s = re.sub(r"\s+", "", str(label))
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        raise CntError(f"숫자를 읽을 수 없는 구간 라벨: {label!r}")
    if "초과" in s and "이하" not in s:
        return float("inf")
    return float(max(nums))


def build_band_labels(labels) -> list:
    keyed = sorted((band_sort_key(x), x) for x in {str(x) for x in labels})
    keys = [k for k, _ in keyed]
    if len(set(keys)) != len(keys):
        raise CntError(f"정렬 키가 겹치는 구간 라벨이 있습니다: {[x for _, x in keyed]}")
    return [x for _, x in keyed]


# --------------------------------------------------------------------------- 패널
@dataclass
class CountPanel:
    months: np.ndarray  # (36,) YYYYMM
    observed: np.ndarray  # (n,36) bool: 그 달 원본 행이 있었는가
    cnt: dict  # 건수 변수 -> int8 (n,36) 구간 코드 0..9, 없으면 -1
    amt: dict  # 금액 변수 -> float64 (n,36), 없으면 NaN (탐지에는 쓰지 않고 결과 검증에만 쓴다)
    attr: dict  # 속성 -> int16 (n,36), 결측 -1
    attr_levels: dict
    band_labels: list
    synthetic: bool = False

    @property
    def n(self) -> int:
        return self.observed.shape[0]

    def save(self, path: Path) -> None:
        arrays = {"months": self.months, "observed": self.observed}
        arrays.update({f"cnt__{k}": v for k, v in self.cnt.items()})
        arrays.update({f"amt__{k}": v for k, v in self.amt.items()})
        arrays.update({f"attr__{k}": v for k, v in self.attr.items()})
        arrays["meta"] = np.array(json.dumps({"attr_levels": self.attr_levels, "band_labels": self.band_labels,
                                              "synthetic": self.synthetic}, ensure_ascii=False))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "CountPanel":
        if not Path(path).exists():
            raise CntError("건수 패널 파일이 없습니다. 먼저 01_표본정의_패널구축 노트북을 실행하세요.")
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(months=z["months"], observed=z["observed"], cnt={c: z[f"cnt__{c}"] for c in CNT_COLS},
                   amt={c: z[f"amt__{c}"] for c in AMT_COLS}, attr={c: z[f"attr__{c}"] for c in ATTR_COLS},
                   attr_levels=meta["attr_levels"], band_labels=meta["band_labels"], synthetic=bool(meta.get("synthetic")))


def build_panel(raw: pd.DataFrame) -> tuple[CountPanel, dict]:
    """원본 행 -> 법인 x 36개월 배열. 어긋나면 고치지 않고 CntError 로 멈춘다."""
    df, rep = raw, {"rows": int(len(raw))}
    ids = df[KEY_ID].astype("string").str.strip()
    if ids.isna().any():
        raise CntError(f"법인ID 결측 {int(ids.isna().sum())}행")
    synthetic = bool(ids.str.startswith("SYN_").mean() > 0.5)
    firm, uniques = pd.factorize(ids)
    n = int(len(uniques))
    del ids, uniques
    firm = firm.astype(np.int64)

    digits = df[KEY_YM].astype("string").str.replace(r"\D", "", regex=True).str[:6]
    ym = pd.to_numeric(digits, errors="coerce")
    if ym.isna().any():
        raise CntError(f"기준년월 변환 실패 {int(ym.isna().sum())}행")
    ym = ym.astype("int64").to_numpy()
    rep["ym_min"], rep["ym_max"] = int(ym.min()), int(ym.max())
    if ym.min() < START_YM or ym.max() > END_YM or ((ym % 100) < 1).any() or ((ym % 100) > 12).any():
        raise CntError(f"기준년월이 {START_YM}~{END_YM} 범위를 벗어났습니다 (실제 {ym.min()}~{ym.max()}).")
    mi = (ym // 100 - START_YM // 100) * 12 + (ym % 100) - 1
    rep["months_present"] = int(np.unique(mi).size)
    dup = int(len(firm) - np.unique(firm * 100 + mi).size)
    rep["dup_pairs"] = dup
    if dup:
        raise CntError(f"(법인ID, 기준년월) 중복 {dup}건. 원인을 확인하기 전에는 진행하지 않습니다.")
    observed = np.zeros((n, T), dtype=bool)
    observed[firm, mi] = True

    amt, fail, neg = {}, {}, {}
    for c in AMT_COLS:
        s = df[c].astype("string").str.replace(",", "", regex=False).str.strip()
        s = s.mask(s == "")
        num = pd.to_numeric(s, errors="coerce").astype("float64")
        fail[c] = int((s.notna() & num.isna()).sum())
        bad = num < 0
        neg[c] = int(bad.sum())
        arr = np.full((n, T), np.nan)
        arr[firm, mi] = num.mask(bad).to_numpy(dtype=np.float64, na_value=np.nan)
        amt[c] = arr
    rep["parse_fail"], rep["negatives"] = fail, neg

    normed = {}
    for c in CNT_COLS:
        s = df[c].astype("string").str.replace(r"\s+", "", regex=True)
        normed[c] = s.mask(s == "")
    union = set()
    for s in normed.values():
        union |= set(s.dropna().unique())
    labels = build_band_labels(union)
    rep["band_labels"] = labels
    if len(labels) != N_BANDS:
        raise CntError(f"거래건수 구간 라벨이 {len(labels)}개입니다 (기대 {N_BANDS}개): {labels}")
    cnt = {}
    for c, s in normed.items():
        codes = pd.Categorical(s, categories=labels).codes.astype(np.int8)
        arr = np.full((n, T), -1, dtype=np.int8)
        arr[firm, mi] = codes
        cnt[c] = arr

    attr, levels = {}, {}
    for c in ATTR_COLS:
        s = df[c].astype("string").str.strip()
        s = s.mask(s == "")
        lv = sorted(set(s.dropna().unique()))
        arr = np.full((n, T), -1, dtype=np.int16)
        arr[firm, mi] = pd.Categorical(s, categories=lv).codes.astype(np.int16)
        attr[c], levels[c] = arr, [str(x) for x in lv]

    panel = CountPanel(months=np.array([(2023 + i // 12) * 100 + (i % 12) + 1 for i in range(T)]), observed=observed, cnt=cnt,
                       amt=amt, attr=attr, attr_levels=levels, band_labels=labels, synthetic=synthetic)
    rep["firms"], rep["synthetic"] = n, synthetic
    return panel, rep


# --------------------------------------------------------------------------- 표본·빈 달 처리
def sample_masks(panel: CountPanel) -> dict:
    """P36: 36개월 완전관측, P35: 정확히 35개월 관측, P35p: 35개월 이상(P36 + P35)."""
    k = panel.observed.sum(axis=1)
    return {"P36": k == 36, "P35": k == 35, "P35p": k >= 35, "nobs": k}


def missing_position(panel: CountPanel, mask: np.ndarray) -> np.ndarray:
    """35개월 관측 기업의 빈 달 위치(0=2023.01 ... 35=2025.12)."""
    obs = panel.observed[mask]
    if (obs.sum(axis=1) != 35).any():
        raise CntError("정확히 35개월 관측인 기업만 넣어야 합니다.")
    return np.argmax(~obs, axis=1)


def codes_float(codes: np.ndarray, observed: np.ndarray | None = None) -> np.ndarray:
    """구간 코드(int8, -1=없음) -> float (없으면 NaN). 규칙 M0(채우지 않음)의 기본 표현."""
    out = codes.astype(np.float64)
    out[codes < 0] = np.nan
    if observed is not None:
        out[~observed] = np.nan
    return out


def fill_interp(codes_f: np.ndarray) -> np.ndarray:
    """규칙 M1: 빈 달을 같은 기업의 관측된 달로 채운다 (중간 공백은 선형 보간 후 반올림, 맨 앞·뒤 공백은 가장 가까운 관측값)."""
    out = codes_f.copy()
    x = np.arange(codes_f.shape[1])
    for i in np.flatnonzero(np.isnan(codes_f).any(axis=1)):
        ok = ~np.isnan(codes_f[i])
        if ok.sum() == 0:
            continue
        v = np.interp(x[~ok], x[ok], codes_f[i, ok])  # 양 끝 밖은 가장 가까운 관측값으로 고정된다
        out[i, ~ok] = np.floor(v + 0.5)
    return out


def approx_count(code_f: np.ndarray, cap: float = DEFAULT_CAP) -> np.ndarray:
    """구간 코드 -> 대표 건수. 코드 9('50건초과')는 열린 구간이라 cap 으로 둔다 (민감도로 바꿔 본다)."""
    mids = np.array(BAND_MIDS + [float(cap)])
    out = np.full(code_f.shape, np.nan)
    m = ~np.isnan(code_f)
    out[m] = mids[code_f[m].astype(int)]
    return out


def two_way_demean(M: np.ndarray, iters: int = 5) -> np.ndarray:
    """기업 평균과 월 평균을 번갈아 빼서 '기업 안에서 시간에 따른 변화'만 남긴다 (NaN 허용)."""
    import warnings

    R = M.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(iters):
            R = R - np.nanmean(R, axis=1, keepdims=True)
            R = R - np.nanmean(R, axis=0, keepdims=True)
    return R


def save_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


# --------------------------------------------------------------------------- Codex 검증 반영 (2026-10-01)
BAND_LOWER = [0, 1, 2, 3, 6, 11, 21, 31, 41, 51]  # 구간 하한 (코드 9 = 50건초과 -> 51)
BAND_UPPER = [0, 1, 2, 5, 10, 20, 30, 40, 50]  # 구간 상한 (코드 9 는 cap_high 로 따로)


def bound_count(code_f: np.ndarray, which: str, cap_high: float = 100.0) -> np.ndarray:
    """구간 코드 -> 구간의 하한 또는 상한 건수. 합산 지수가 구간 안 분포 가정에 얼마나 민감한지 보는 민감도용."""
    vals = np.array(BAND_LOWER, dtype=float) if which == "low" else np.array(BAND_UPPER + [float(cap_high)])
    out = np.full(code_f.shape, np.nan)
    m = ~np.isnan(code_f)
    out[m] = vals[code_f[m].astype(int)]
    return out


def paired_means(M: np.ndarray, e: int, k: int, need: int):
    """쌍 맞춤 비교(M0): 최근 k개월 창 [e-k+1 .. e] 과 1년 전 같은 달 창에서 **두 해 모두 관측된 달만** 짝지어 평균한다.

    반환: (최근 평균, 1년 전 평균) 각각 (n,), 짝지은 달이 need 개 미만이면 NaN.
    '같은 k개월'을 비교한다는 전제가 빈 달 때문에 깨지지 않도록 하기 위한 규칙이다. e >= k + 11 이어야 한다.
    """
    import warnings

    a = e - k + 1
    if a - 12 < 0:
        raise ValueError("1년 전 같은 달 창이 패널 밖입니다 (e >= k + 11 이어야 함)")
    cur, old = M[:, a : e + 1], M[:, a - 12 : e + 1 - 12]
    ok = ~np.isnan(cur) & ~np.isnan(old)
    cnt = ok.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rc = np.nanmean(np.where(ok, cur, np.nan), axis=1)
        pr = np.nanmean(np.where(ok, old, np.nan), axis=1)
    return np.where(cnt >= need, rc, np.nan), np.where(cnt >= need, pr, np.nan)


# --------------------------------------------------------------------------- Codex 2차 검증 반영 (2026-10-01)
def min_pairs(k: int) -> int:
    """M0 쌍 맞춤에서 필요한 최소 짝지은 달 수. k=1 이어도 최소 1쌍은 있어야 한다 (k-1=0 이면 아무 것도 비교하지 않게 됨)."""
    return max(int(k) - 1, 1)


def fill_causal(codes_f: np.ndarray, s: int) -> np.ndarray:
    """M1(인과적 채움): 후보 시점 s 에서 **실제로 이용 가능했던 값(열 <= s)만** 으로 채운 배열.

    - 열 > s 는 NaN (그 시점에는 아직 알 수 없는 값).
    - 열 s 자체가 미관측인 기업은 행 전체 NaN (판정 불가).
    - s 이전의 빈 달은 s 까지의 관측값으로 선형 보간 후 반올림하고, 첫 관측 이전은 가장 가까운 이후 관측값(<= s)으로 채운다.
    탐지 이력은 후보월마다 이 함수를 다시 적용해 순차적으로 재계산해야 한다 (s 이후 값으로 과거를 보간하면 미래 정보가 섞인다).
    """
    n, T = codes_f.shape
    out = np.full(codes_f.shape, np.nan)
    x = np.arange(s + 1)
    for i in range(n):
        row = codes_f[i, : s + 1]
        if np.isnan(row[s]):
            continue
        ok = ~np.isnan(row)
        v = row.copy()
        if (~ok).any():
            v[~ok] = np.floor(np.interp(x[~ok], x[ok], row[ok]) + 0.5)
        out[i, : s + 1] = v
    return out
