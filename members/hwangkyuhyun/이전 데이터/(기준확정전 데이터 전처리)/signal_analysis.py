import pandas as pd
import numpy as np
from IPython.display import display


# ==========================================================
# 0. 데이터 불러오기
# ==========================================================

df = pd.read_excel("im_bank_data.xlsx")

# 현재 데이터는 기존 구조 검증이 끝났다고 가정
clean_panel = df.copy()


# ==========================================================
# 1. 분석에 필요한 변수만 추출
# ==========================================================

signal_base = clean_panel[
    ["법인ID", "기준년월", "요구불입금금액"]
].copy()

# YYYYMM → 월 단위 Period로 변환
signal_base["월"] = pd.to_datetime(
    signal_base["기준년월"].astype(str),
    format="%Y%m"
).dt.to_period("M")


# ==========================================================
# 2. 법인 × 월 패널 생성
# ==========================================================

# 실제 요구불입금금액
amount_panel = signal_base.pivot(
    index="법인ID",
    columns="월",
    values="요구불입금금액"
)

# 해당 법인이 해당 월에 실제로 관측됐는지 확인
presence_base = signal_base[
    ["법인ID", "월"]
].copy()

presence_base["관측"] = True

presence_panel = (
    presence_base
    .pivot(
        index="법인ID",
        columns="월",
        values="관측"
    )
    .fillna(False)
)


# ==========================================================
# 3. 분석 기준월 설정
# ==========================================================
# T-14가 존재하고,
# 이후 T+6까지 추적 가능한 기간
# ==========================================================

reference_months = pd.period_range(
    "2024-03",
    "2025-06",
    freq="M"
)


# ==========================================================
# 4. Case 1 / Case 2 위축신호 계산
# ==========================================================

signal_event_list = []

for T in reference_months:

    # 전년 기준 3개월
    before_months = [
        T - 14,
        T - 13,
        T - 12
    ]

    # 현재 기준 3개월
    after_months = [
        T - 2,
        T - 1,
        T
    ]

    # ------------------------------------------
    # 실제 관측 개월 수
    # ------------------------------------------

    before_n = presence_panel[
        before_months
    ].sum(axis=1)

    after_n = presence_panel[
        after_months
    ].sum(axis=1)


    # ------------------------------------------
    # T와 T-12는 반드시 실제 관측
    # ------------------------------------------

    anchor_ok = (
        presence_panel[T]
        & presence_panel[T - 12]
    )


    # ------------------------------------------
    # 각 3개월 창의 대표값 = 중앙값
    # ------------------------------------------

    before_median = amount_panel[
        before_months
    ].median(
        axis=1,
        skipna=True
    )

    after_median = amount_panel[
        after_months
    ].median(
        axis=1,
        skipna=True
    )


    # ======================================================
    # Case 1
    # - T, T-12 실제 관측
    # - before / after 각각 최소 2개월 이상 관측
    # ======================================================

    case1_obs_ok = (
        anchor_ok
        & (before_n >= 2)
        & (after_n >= 2)
    )

    # 전년 중앙값이 0이면 위축률 계산 불가
    case1_calc_ok = (
        case1_obs_ok
        & (before_median > 0)
    )

    # 위축률
    case1_rate = (
        (before_median - after_median)
        / before_median
    )

    # 30% 이상 감소
    case1_signal = (
        case1_calc_ok
        & (case1_rate >= 0.30)
    )


    # ======================================================
    # Case 2
    # - before 3/3 관측
    # - after 3/3 관측
    # ======================================================

    case2_obs_ok = (
        anchor_ok
        & (before_n == 3)
        & (after_n == 3)
    )

    case2_calc_ok = (
        case2_obs_ok
        & (before_median > 0)
    )

    case2_signal = (
        case2_calc_ok
        & (case1_rate >= 0.30)
    )


    # ------------------------------------------
    # 법인 × 기준월 단위 결과 저장
    # ------------------------------------------

    tmp = pd.DataFrame({

        "법인ID":
            amount_panel.index,

        "기준월":
            str(T),

        "before_관측개월수":
            before_n.values,

        "after_관측개월수":
            after_n.values,

        "before_중앙값":
            before_median.values,

        "after_중앙값":
            after_median.values,

        "위축률":
            case1_rate.values,

        "Case1_관측조건":
            case1_obs_ok.values,

        "Case1_계산가능":
            case1_calc_ok.values,

        "Case1_30퍼신호":
            case1_signal.values,

        "Case2_관측조건":
            case2_obs_ok.values,

        "Case2_계산가능":
            case2_calc_ok.values,

        "Case2_30퍼신호":
            case2_signal.values
    })

    signal_event_list.append(tmp)


# 모든 기준월 결과 합치기
signal_events = pd.concat(
    signal_event_list,
    ignore_index=True
)


# ==========================================================
# 5. Case 1 / Case 2 기준월별 요약
# ==========================================================

signal_summary = (
    signal_events
    .groupby("기준월")
    .agg(

        Case1_관측조건충족=(
            "Case1_관측조건",
            "sum"
        ),

        Case1_계산가능=(
            "Case1_계산가능",
            "sum"
        ),

        Case1_30퍼신호=(
            "Case1_30퍼신호",
            "sum"
        ),

        Case2_완전관측=(
            "Case2_관측조건",
            "sum"
        ),

        Case2_계산가능=(
            "Case2_계산가능",
            "sum"
        ),

        Case2_30퍼신호=(
            "Case2_30퍼신호",
            "sum"
        )
    )
    .reset_index()
)


# Case 1 대비 Case 2에서 얼마나 표본이 유지되는지
signal_summary["Case2_표본유지율"] = (
    signal_summary["Case2_계산가능"]
    / signal_summary["Case1_계산가능"]
)

display(signal_summary)


# ==========================================================
# 6. 위축 임계값 민감도 분석
# 20% / 30% / 40%
# ==========================================================

thresholds = {
    "20퍼": 0.20,
    "30퍼": 0.30,
    "40퍼": 0.40
}


# ------------------------------------------
# Case 1 / Case 2별 임계값 신호 생성
# ------------------------------------------

for name, threshold in thresholds.items():

    signal_events[f"Case1_{name}신호"] = (
        signal_events["Case1_계산가능"]
        & (
            signal_events["위축률"]
            >= threshold
        )
    )

    signal_events[f"Case2_{name}신호"] = (
        signal_events["Case2_계산가능"]
        & (
            signal_events["위축률"]
            >= threshold
        )
    )


# ==========================================================
# 7. 기준월별 임계값 민감도 요약
# ==========================================================

threshold_summary = (
    signal_events
    .groupby("기준월")
    .agg(

        Case1_계산가능=(
            "Case1_계산가능",
            "sum"
        ),

        Case1_20퍼=(
            "Case1_20퍼신호",
            "sum"
        ),

        Case1_30퍼=(
            "Case1_30퍼신호",
            "sum"
        ),

        Case1_40퍼=(
            "Case1_40퍼신호",
            "sum"
        ),

        Case2_계산가능=(
            "Case2_계산가능",
            "sum"
        ),

        Case2_20퍼=(
            "Case2_20퍼신호",
            "sum"
        ),

        Case2_30퍼=(
            "Case2_30퍼신호",
            "sum"
        ),

        Case2_40퍼=(
            "Case2_40퍼신호",
            "sum"
        )
    )
    .reset_index()
)


# ------------------------------------------
# 계산가능 표본 중 위축신호 비율
# ------------------------------------------

for case in [
    "Case1",
    "Case2"
]:

    for pct in [
        "20퍼",
        "30퍼",
        "40퍼"
    ]:

        threshold_summary[
            f"{case}_{pct}_신호율"
        ] = (

            threshold_summary[
                f"{case}_{pct}"
            ]

            /

            threshold_summary[
                f"{case}_계산가능"
            ]
        )


display(threshold_summary)