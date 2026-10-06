"""발표 6~8장 그림. 지정된 집계 CSV만 읽어 PNG/SVG와 사용 값 표를 만든다."""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
WORK = Path(os.environ.get("CNT_WORK_DIR") or ROOT / "data" / "work_cnt")
SOURCE = WORK / "output"
DEST = SOURCE / "fig_6_8"

INK = "#192338"
NAVY = "#1E2E4F"
BLUE = "#31487A"
SKY = "#8FB3E2"
LAV = "#D9E1F1"
ORANGE = "#E09F3E"
GRAY = "#9AA5B8"
MUTED = "#5B6478"
GRID = "#E6EAF2"

VALUE_ROWS: list[dict[str, str]] = []
FIGURES: list[tuple[str, Path]] = []
CHECKS: list[tuple[str, bool]] = []


def read_csv(relative: str) -> list[dict[str, str]]:
    with (SOURCE / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | None:
    value = str(value).strip().replace(",", "")
    if value in {"", "-", "<5"}:
        return None
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"숫자를 읽을 수 없습니다: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"유한한 숫자가 아닙니다: {value!r}")
    return result


TRIPLE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*[~～]\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$")
PAIR = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*[~～]\s*([+-]?\d+(?:\.\d+)?)\s*$")


def interval(value: str, estimate: str | float | None = None) -> tuple[float, float, float]:
    """'추정 (하한~상한)'과 '하한 ~ 상한'을 한 함수로 검사한다."""
    value = str(value).strip()
    match = TRIPLE.fullmatch(value)
    if match:
        mid, low, high = map(float, match.groups())
    else:
        match = PAIR.fullmatch(value)
        if match is None or estimate is None:
            raise ValueError(f"95% 범위를 파싱할 수 없습니다: {value!r}")
        low, high = map(float, match.groups())
        mid = number(estimate) if isinstance(estimate, str) else estimate
        if mid is None:
            raise ValueError(f"추정값이 없습니다: {value!r}")
    if not (math.isfinite(mid) and math.isfinite(low) and math.isfinite(high) and low <= mid <= high):
        raise ValueError(f"95% 범위의 순서가 잘못되었습니다: {value!r}")
    return mid, low, high


def add_value(fig: str, item: str, estimate: float | None, low: float | None, high: float | None,
              source: str, note: str = "") -> None:
    def cell(x: float | None) -> str:
        return "" if x is None else f"{x:g}"
    VALUE_ROWS.append({"그림": fig, "항목": item.replace("−", "-"), "추정": cell(estimate), "하한": cell(low),
                       "상한": cell(high), "출처 파일": source, "비고": note.replace("−", "-")})


def configure() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        raise RuntimeError("사용 가능한 한글 글꼴이 없습니다")
    plt.rcParams.update({
        "axes.unicode_minus": False, "font.size": 14, "axes.labelsize": 15,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none", "svg.hashsalt": "fig_6_8",
    })


def style(ax, axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.tick_params(length=0, pad=8)


def rr_axis(ax, top: float) -> None:
    """배수는 1이 '차이 없음'이므로 축 바닥을 1로 둔다(막대·점의 높이 = 1에서 떨어진 정도)."""
    style(ax)
    ax.set_ylim(1, top)
    ax.set_yticks(range(1, int(top) + 1))
    ax.axhline(1, color=GRAY, linewidth=1.8, zorder=1, clip_on=False)
    ax.text(0.01, 0.015, "1 = 차이 없음", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=14, color=MUTED)
    ax.set_ylabel("배수")


def ci_mark(ax, x: float, triple: tuple[float, float, float], color: str = NAVY,
            label: bool = True, label_dx: float = 0.10) -> None:
    mid, low, high = triple
    ax.errorbar([x], [mid], yerr=[[mid - low], [high - mid]], fmt="o", color=color, markersize=9,
                elinewidth=2.2, capsize=10, capthick=2.2, zorder=4)
    if label:
        ax.annotate(f"{mid:.2f}", (x, mid), xytext=(8, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=14, fontweight="bold", color=INK)


def ci_bracket(ax, x: float, triple: tuple[float, float, float], dx: float = 0.13) -> None:
    """오차막대 왼쪽에 회색 괄호와 '95% 신뢰구간' 글자 (범례 대신 직접 표시)."""
    _, low, high = triple
    ax.plot([x - dx + 0.05, x - dx, x - dx, x - dx + 0.05], [high, high, low, low], color=GRAY, linewidth=1.4)
    ax.text(x - dx - 0.04, (low + high) / 2, "95%\n신뢰구간", ha="right", va="center", fontsize=13,
            color=MUTED, linespacing=1.3)


def save(fig, stem: str, axes: list | tuple, rr_axes: list = (), orange_groups: int = 0) -> None:
    for ax in rr_axes:
        ylim = ax.get_ylim()
        line_found = any(isinstance(line, Line2D) and line.get_color() == GRAY and
                         len(line.get_ydata()) == 2 and list(line.get_ydata()) == [1, 1]
                         for line in ax.lines)
        CHECKS.append((f"{stem}: 배수 축 1(차이 없음) 시작", abs(ylim[0] - 1) < 1e-9 and line_found))
    CHECKS.append((f"{stem}: 주황 강조 한 곳 이하", orange_groups <= 1))
    fig.savefig(DEST / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(DEST / f"{stem}.svg", bbox_inches="tight", pad_inches=0.18,
                metadata={"Date": None})
    FIGURES.append((stem, DEST / f"{stem}.png"))
    plt.close(fig)


def jitter(values: list[float], step: float = 0.035, gap: float = 0.21, levels: int = 6) -> np.ndarray:
    """겹치는 점만 좌우로 비켜 놓는다(값 순서와 무관한 결정적 배치)."""
    candidates = [0.0] + [sign * k * step for k in range(1, levels + 1) for sign in (1, -1)]
    placed: list[tuple[float, float]] = []
    result = np.zeros(len(values))
    for i in np.argsort(values, kind="stable"):
        y = values[i]
        for c in candidates:
            if all(abs(c - o) >= step * 0.99 or abs(y - py) >= gap for o, py in placed):
                break
        placed.append((c, y))
        result[i] = c
    return result


def distribution(ax, values: list[float], color: str = NAVY) -> np.ndarray:
    ax.boxplot([values], positions=[0], widths=0.30, patch_artist=True, showfliers=False,
               boxprops={"facecolor": LAV, "edgecolor": color, "linewidth": 1.8},
               medianprops={"color": color, "linewidth": 2.2},
               whiskerprops={"color": color, "linewidth": 1.5},
               capprops={"color": color, "linewidth": 1.5})
    xs = jitter(values)
    ax.scatter(xs, values, s=52, color=color, edgecolor="white", linewidth=0.7, zorder=4)
    ax.set_xlim(-0.42, 0.42)
    return xs


def fig_6_2() -> None:
    src = "nb05/hc1_main.csv"
    data = {row["항목"]: row["값"] for row in read_csv(src)}
    signal = number(data["탐지군 약화율 %"])
    comparison = number(data["표준화 대조군 약화율 %"])
    n = number(data["탐지군 입금 약화 건수"])
    total = number(data["탐지 기업 N1"])
    rr, rr_low, rr_high = interval(data["95% CI"], data["RR"])
    if any(v is None for v in (signal, comparison, n, total)):
        raise ValueError(f"필수 집계값이 없습니다: {src}")
    add_value("6-2", "신호 기업 관계 위축 비율 (%)", signal, None, None, src)
    add_value("6-2", "비교 기업 관계 위축 비율 (%)", comparison, None, None, src)
    add_value("6-2", "관계 위축 기업 (곳)", n, None, None, src)
    add_value("6-2", "신호 기업 전체 (곳)", total, None, None, src)
    add_value("6-2", "배수", rr, rr_low, rr_high, src)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    # 세로 눈금(0~30%, 5% 간격)을 보여 막대 높이를 그대로 읽게 함
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRAY)
        ax.spines[side].set_linewidth(1.3)
    ax.set_ylim(0, 31)
    ax.set_yticks(range(0, 31, 5), [f"{v}%" for v in range(0, 31, 5)])
    ax.tick_params(axis="y", length=0, labelsize=14, labelcolor=MUTED, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 2.15)
    ax.bar([0, 1], [signal, comparison], width=0.52, color=[NAVY, SKY], zorder=2)
    # 막대 위: 큰 값 + 작은 설명
    ax.text(0, signal + 1.0, f"{signal:.1f}%", ha="center", va="bottom", fontsize=30, fontweight="bold", color=INK)
    ax.text(1, comparison + 1.0, f"{comparison:.1f}%", ha="center", va="bottom", fontsize=30, fontweight="bold", color=INK)
    # 막대 아래: 큰 이름 + 작은 설명
    ax.set_xticks([0, 1], ["", ""])
    ax.tick_params(axis="x", length=0)
    ax.text(0, -2.2, "신호 기업", ha="center", va="top", fontsize=20, fontweight="bold", color=NAVY)
    ax.text(1, -2.2, "비교 기업", ha="center", va="top", fontsize=20, fontweight="bold", color=BLUE)
    # 위쪽 가로 설명 (세로 축 이름 대신)
    # 주황 괄호 + 배수 + 신뢰구간
    bx = 1.42
    ax.plot([bx - 0.08, bx, bx, bx - 0.08], [signal, signal, comparison, comparison], color=ORANGE, linewidth=2.6,
            solid_joinstyle="miter")
    mid = (signal + comparison) / 2
    ax.text(bx + 0.09, mid + 6.4, "상대위험도(RR)", ha="left", va="bottom", fontsize=15, fontweight="bold", color=MUTED)
    ax.text(bx + 0.08, mid + 1.6, f"{rr:.1f}배", ha="left", va="bottom", fontsize=34, fontweight="bold", color=ORANGE)
    ax.text(bx + 0.09, mid - 0.4, f"95% 신뢰구간 {rr_low:.2f}~{rr_high:.2f}배", ha="left", va="top", fontsize=13, color=MUTED)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)
    save(fig, "fig_6-2_비율비교", [ax], orange_groups=1)


def fig_6_3() -> None:
    left_src, right_src = "nb05/holdout_main.csv", "nb05/leave_one_month.csv"
    holdout = read_csv(left_src)[:2]
    excluded = read_csv(right_src)
    triples = [interval(row["RR v2 (95% CI)"]) for row in holdout]
    values = [number(row["RR"]) for row in excluded]
    if any(v is None for v in values):
        raise ValueError(f"가려진 값은 그림에 넣을 수 없습니다: {right_src}")
    for row, triple in zip(holdout, triples):
        add_value("6-3", row["표본"], *triple, left_src)
    for row, value in zip(excluded, values):
        add_value("6-3", f"제외 기준월 {row['제외한 기준월']}", value, None, None, right_src)
    fig, (a, b) = plt.subplots(1, 2, sharey=True, figsize=(12.5, 5.4),
                               gridspec_kw={"width_ratios": [0.9, 1.6]})
    rr_axis(a, 8)
    rr_axis(b, 8)
    # 왼쪽: 서로 다른(겹치지 않는) 기업 표본 2개
    a.set_xlim(-0.75, 1.5)
    a.set_xticks(range(2), ["36개월 기업\n(표본)", "35개월 기업\n(독립 검증)"])
    for x, triple in enumerate(triples):
        ci_mark(a, x, triple)
    ci_bracket(a, 0, triples[0])
    a.set_title("독립 검증 (홀드아웃)", loc="left", fontsize=15, color=INK, fontweight="bold", pad=12)
    # 오른쪽: 기준월을 한 달씩 빼고 다시 계산한 값(점추정). 시간 추세로 읽히지 않게 선으로 잇지 않고,
    # 전체 결과의 95% 신뢰구간을 띠로 깔아 '빼도 움직인 폭'을 '전체 불확실성'과 견주게 함
    months = [str(int(row["제외한 기준월"])) for row in excluded]
    xs = list(range(len(values)))
    # 띠 = 36개월 표본의 95% 신뢰구간. 띠 바로 위 흰 바탕에 이름과 범위를 적음
    b.set_xlim(-0.6, len(values) - 0.4)
    _, low0, high0 = triples[0]
    b.axhspan(low0, high0, facecolor="#E3E9F5", edgecolor="none", zorder=0)
    b.text(-0.45, high0 + 0.08, f"36개월 표본의 95% 신뢰구간 ({low0:.2f}~{high0:.2f})", ha="left", va="bottom",
           fontsize=13, color=BLUE, zorder=1)
    b.axhline(triples[0][0], color=ORANGE, linestyle=(0, (5, 3)), linewidth=2, zorder=2)
    b.text(len(values) - 0.5, triples[0][0] + 0.5, f"전체 기준월 사용 {triples[0][0]:.2f}",
           ha="right", va="bottom", color=ORANGE, fontsize=14)
    b.scatter(xs, values, s=70, color=NAVY, edgecolor="white", linewidth=1, zorder=4)
    b.set_xticks(xs, [f"{m[2:4]}.{m[4:]}" for m in months], fontsize=12,
                 rotation=45, ha="right", rotation_mode="anchor")
    b.set_ylabel("")
    b.set_title("Leave-one-out 민감도 분석 (기준월)", loc="left", fontsize=15, color=INK, fontweight="bold", pad=12)
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig_6-3_재현", [a, b], [a, b], orange_groups=1)


def fig_6_4() -> None:
    path_src, sum_src = "nb05/hc2_path.csv", "nb05/hc2_summary.csv"
    paths = read_csv(path_src)
    summary = read_csv(sum_src)
    months = [int(row["상대월 j"]) for row in paths]
    deltas = [number(row["Δ(j)"]) for row in paths]
    if any(v is None for v in deltas):
        raise ValueError(f"가려진 월별 값이 있습니다: {path_src}")
    for row, value in zip(paths, deltas):
        add_value("6-4a / 6-4b", f"상대월 {row['상대월 j']}", value, None, None, path_src)
    selected = [summary[i] for i in (0, 2, 3)]
    means = [interval(row["95% CI"], row["점추정"]) for row in selected]
    for row, triple in zip(selected, means):
        add_value("6-4b", row["요약"], *triple, sum_src)
    limits = (min(-0.5, min(deltas) - 0.2), max(deltas) + 0.8)
    spans = [(-12.5, -5.5, "거래 횟수 줄기 전", LAV, 0.48),
             (-5.5, 0.5, "거래 횟수 줄어드는 중", SKY, 0.30),
             (0.5, 6.5, "신호 이후", ORANGE, 0.15)]
    def common(ax):
        style(ax)
        ax.set_xlim(-12.5, 6.5)
        ax.set_ylim(*limits)
        for lo, hi, name, color, alpha in spans:
            ax.axvspan(lo, hi, facecolor=color, alpha=alpha, zorder=0)
            ax.text((lo + hi) / 2, 0.98, name, ha="center", va="top", transform=ax.get_xaxis_transform(),
                    fontsize=14, color=MUTED)
        ax.axhline(0, color=INK, linewidth=2, zorder=3)
        ax.text(-12.3, -0.08, "0 = 두 집단의 변화가 같음", ha="left", va="top", fontsize=14, color=INK)
        ax.set_xticks([-12, -9, -6, -3, 0, 3, 6])
        ax.set_xlabel("신호 기준 상대월")
        ax.set_ylabel("작년 대비 입금 비율의 차이\n(신호 기업 - 비교 기업)")
    fig, ax = plt.subplots(figsize=(12, 5))
    common(ax)
    ax.plot(months, deltas, color=NAVY, linewidth=2.3, marker="o", markersize=7, zorder=5)
    fig.text(0.5, 0.015, "월별 값은 95% 범위가 매우 넓어 흐름만 봄(탐색 분석)",
             ha="center", color=MUTED, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig_6-4a_신호전후", [ax], orange_groups=1)

    fig, ax = plt.subplots(figsize=(12, 5))
    common(ax)
    ax.scatter(months, deltas, s=35, color=GRAY, zorder=4)
    for (lo, hi, _, _, _), (mid, lower, upper) in zip(spans, means):
        upper_clip = min(upper, limits[1])
        ax.add_patch(Rectangle((lo + 0.16, max(lower, limits[0])), hi - lo - 0.32,
                               upper_clip - max(lower, limits[0]), facecolor=LAV,
                               edgecolor="none", alpha=0.52, zorder=1))
        if upper > limits[1]:
            ax.annotate("", xy=((lo + hi) / 2, limits[1] - 0.07),
                        xytext=((lo + hi) / 2, limits[1] - 0.55),
                        arrowprops={"arrowstyle": "-|>", "color": NAVY, "lw": 1.6})
        ax.plot([lo + 0.18, hi - 0.18], [mid, mid], color=NAVY, linewidth=5,
                solid_capstyle="round", zorder=6)
        ax.text((lo + hi) / 2, mid + 0.13, f"{mid:.2f}", ha="center", va="bottom",
                color=INK, fontsize=14, fontweight="bold", zorder=7)
    fig.tight_layout()
    save(fig, "fig_6-4b_신호전후", [ax], orange_groups=1)


def fig_7_1() -> None:
    src = "nb05/heterogeneity.csv"
    rows = read_csv(src)[:5]
    rates = [number(row["기업별 첫 탐지율 %"]) for row in rows]
    counts = [number(row["탐지 기업"]) for row in rows]
    events = [number(row["약화 사건(탐지군)"]) for row in rows]
    triples = [interval(row["RR (95% CI)"]) if row["RR (95% CI)"].strip() != "<5" else None for row in rows]
    shows = []
    for i, (rate, count, event, triple) in enumerate(zip(rates, counts, events, triples), 1):
        note = "<5: 집계값 가림" if None in (rate, count, event) else ""
        add_value("7-1", f"{i}분위 신호 비율 (%)", rate, None, None, src, note if rate is None else "")
        add_value("7-1", f"{i}분위 신호 기업 (곳)", count, None, None, src, note if count is None else "")
        add_value("7-1", f"{i}분위 관계 위축 (건)", event, None, None, src, note if event is None else "")
        show = count is not None and event is not None and count >= 30 and event > 5 and triple is not None
        shows.append(show)
        if triple is not None:
            add_value("7-1", f"{i}분위 배수", *triple, src, "크기 해석 제외" if not show else "")
    # 한 그림: 거래량 단계별로 신호가 잡힌 비율(막대). 아래 줄: 신호가 잡혔을 때의 상대위험도
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    style(ax)
    ax.set_ylim(0, 50)
    ax.set_yticks(range(0, 51, 10), [f"{v}%" for v in range(0, 51, 10)])
    ax.set_xlim(-0.6, 4.6)
    available = [i for i, rate in enumerate(rates) if rate is not None]
    lowest = set(sorted(available, key=lambda i: rates[i])[:2])
    ax.bar(available, [rates[i] for i in available], width=0.56, zorder=2,
           color=[ORANGE if i in lowest else NAVY for i in available])
    for x, value in enumerate(rates):
        if value is None:
            ax.text(x, 2, "<5: 집계값 가림", ha="center", color=MUTED, fontsize=13)
        else:
            ax.text(x, value + 1, f"{value:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=17,
                    color=INK)
    if len(lowest) == 2:
        lo, hi = min(lowest), max(lowest)
        top = max(rates[i] for i in lowest) + 8
        ax.plot([lo, lo, hi, hi], [top - 2, top, top, top - 2], color=ORANGE, linewidth=2)
        ax.text((lo + hi) / 2, top + 1, "사각지대", ha="center", color=ORANGE, fontweight="bold", fontsize=16)
    ax.set_xticks(range(5), ["1분위\n거래 적음", "2분위", "3분위", "4분위", "5분위\n거래 많음"])
    ax.tick_params(axis="x", labelsize=14, labelcolor=INK)
    ax.set_ylabel("신호 기업 비율")
    # 아래 줄 (x축 이름 아래)
    trans = ax.get_xaxis_transform()
    row_y = -0.33
    ax.plot([-0.6, 4.6], [row_y + 0.10, row_y + 0.10], transform=trans, clip_on=False, color=GRID, linewidth=1.2)
    ax.text(-0.75, row_y, "상대위험도(RR)", transform=trans, ha="right", va="center",
            fontsize=13, color=MUTED, linespacing=1.3)
    for x, (count, triple, show) in enumerate(zip(counts, triples, shows)):
        if show:
            ax.text(x, row_y, f"{triple[0]:.1f}배", transform=trans, ha="center", va="center", fontsize=16,
                    fontweight="bold", color=NAVY)
        else:
            label = "집계값 가림" if count is None else f"{int(count)}곳뿐이라\n생략"
            ax.text(x, row_y, label, transform=trans, ha="center", va="center", fontsize=12, color=MUTED,
                    linespacing=1.2)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.95, bottom=0.26)
    save(fig, "fig_7-1_거래량별", [ax], orange_groups=1)


def fig_7_2() -> None:
    src = "nb06/c2_type_rr.csv"
    rows = read_csv(src)
    triples = [interval(row["배수(RR) (95% 범위)"]) if row["배수(RR) (95% 범위)"].strip() != "<5" else None for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    rr_axis(ax, 12)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3), ["인터넷·스마트\n중심형", "혼합형", "창구·ATM\n중심형"])
    for x, (row, triple) in enumerate(zip(rows, triples)):
        if triple is None:
            ax.text(x, 5, "<5: 집계값 가림\n그림에서 제외", ha="center", color=MUTED, fontsize=14)
            add_value("7-2", row["집단"], None, None, None, src, "<5: 집계값 가림")
        else:
            ci_mark(ax, x, triple, color=BLUE)
            add_value("7-2", row["집단"], *triple, src)
    fig.tight_layout()
    save(fig, "fig_7-2_채널유형", [ax], [ax])


def shorten(name: str) -> str:
    name = re.sub(r"\s*\([^)]*\)", "", name).strip()
    return name if len(name) <= 22 else name[:21] + "…"


# 8-1 묶음: CSV 의 '묶음' → 발표용 이름 (활동량 하한·n·k 는 둘 다 신호 규칙이라 합침)
GROUPS_8_1 = [("거래 횟수\n세는 방식", {"구간 가정"}),
              ("신호 규칙\n(문턱·비교 기간)", {"활동량 하한", "n·k"}),
              ("관계 위축\n정의", {"입금 약화 정의"}),
              ("비교 기업\n구성", {"비교군·층"}),
              ("표본·기간", {"표본·기간"})]


def fig_8_1() -> None:
    src = "nb07/variants.csv"
    variants = read_csv(src)
    bases = [row for row in variants if row["묶음"] == "기준"]
    changed = [row for row in variants if row["묶음"] != "기준"]
    if len(bases) != 1 or len(changed) != 20:
        raise ValueError(f"기준 1줄과 변형 20줄이 필요합니다: {src}")
    base = interval(bases[0]["RR (95% 범위)"])
    add_value("8-1", "본 분석 (기준)", *base, src)
    known = set().union(*(g for _, g in GROUPS_8_1))
    if {row["묶음"] for row in changed} - known:
        raise ValueError(f"묶음 이름이 예상과 다릅니다: {sorted({row['묶음'] for row in changed} - known)}")
    fig, ax = plt.subplots(figsize=(12, 5.6))
    rr_axis(ax, 10)
    gap, step = 1.4, 0.55   # 묶음 사이 간격, 묶음 안 점 간격
    x, centers, all_low = 0.0, [], []
    for gi, (label, keys) in enumerate(GROUPS_8_1):
        rows = [row for row in changed if row["묶음"] in keys]
        triples = sorted((interval(row["RR (95% 범위)"]), row["변형"]) for row in rows)
        xs = [x + k * step for k in range(len(triples))]
        for xi, ((mid, low, high), name) in zip(xs, triples):
            add_value("8-1", name, mid, low, high, src)
            all_low.append(low)
            ax.errorbar([xi], [mid], yerr=[[mid - low], [high - mid]], fmt="o", color=NAVY, ecolor=SKY,
                        markersize=8, elinewidth=2.4, capsize=5, capthick=2.0, zorder=3)
        centers.append((xs[0] + xs[-1]) / 2)
        if gi < len(GROUPS_8_1) - 1:
            ax.axvline(xs[-1] + gap / 2, color=GRID, linewidth=1.2, zorder=0)
        x = xs[-1] + gap
    if not all(low > 1 for low in all_low):
        print("WARNING: 20가지 변형 중 95% 범위 하한이 1 이하인 항목이 있습니다")
    ax.set_xlim(-1.5, x - gap + 1.9)
    ax.set_xticks(centers, [label for label, _ in GROUPS_8_1])
    ax.tick_params(axis="x", labelsize=14, labelcolor=INK)
    for c, (_, keys) in zip(centers, GROUPS_8_1):
        n = sum(row["묶음"] in keys for row in changed)
        ax.text(c, -0.15, f"{n}가지", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=12,
                color=MUTED)
    # 본 분석 점선은 마지막 묶음 바로 뒤에서 끊고, 그 오른쪽에 이름을 둠 (글자 위로 선이 지나가지 않게)
    ax.hlines(base[0], -1.5, x - gap + 0.35, color=ORANGE, linestyle=(0, (5, 3)), linewidth=2, zorder=1)
    ax.text(x - gap + 0.45, base[0], f"본 분석\n{base[0]:.2f}", ha="left", va="center", color=ORANGE,
            fontsize=14, linespacing=1.2)
    first = sorted(interval(row["RR (95% 범위)"]) for row in changed if row["묶음"] in GROUPS_8_1[0][1])[0]
    ci_bracket(ax, 0, first, dx=0.22)
    ax.set_ylabel("상대위험도(RR)")
    fig.tight_layout()
    save(fig, "fig_8-1_기준변경", [ax], [ax], orange_groups=1)


def overview() -> None:
    thumbs = []
    font_path = font_manager.findfont(plt.rcParams["font.family"][0])
    font = ImageFont.truetype(font_path, 25)
    for label, path in FIGURES:
        with Image.open(path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((900, 480))
            tile = Image.new("RGB", (940, 540), "white")
            tile.paste(thumb, ((940 - thumb.width) // 2, 42 + (480 - thumb.height) // 2))
            ImageDraw.Draw(tile).text((20, 8), label, font=font, fill=INK)
            thumbs.append(tile)
    sheet = Image.new("RGB", (940 * 2, 540 * math.ceil(len(thumbs) / 2)), GRID)
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 2) * 940, (i // 2) * 540))
    sheet.save(DEST / "overview.png")


def write_values() -> None:
    path = DEST / "values_used.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["그림", "항목", "추정", "하한", "상한", "출처 파일", "비고"])
        writer.writeheader()
        writer.writerows(VALUE_ROWS)
    print("그림마다 사용한 집계값")
    print("그림 | 항목 | 추정 | 하한 | 상한 | 출처 파일 | 비고")
    for row in VALUE_ROWS:
        print(" | ".join(row.values()))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    configure()
    DEST.mkdir(parents=True, exist_ok=True)
    fig_6_2()
    fig_6_3()
    fig_6_4()
    fig_7_1()
    fig_7_2()
    fig_8_1()
    overview()
    write_values()
    for name, passed in CHECKS:
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    if not all(passed for _, passed in CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
