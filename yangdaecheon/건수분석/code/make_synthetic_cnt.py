"""합성(난수) 테스트 데이터 -- 건수분석용. 실제 데이터와 무관하다.

원본 명세서와 같은 열 이름·형식(구간 라벨, 반올림된 금액, 행이 없는 달)을 흉내 내고, 분석 코드가 신호를 제대로 찾는지
확인할 수 있도록 알려진 구조를 일부러 심어 둔다 (심어 둔 것 = 정답).
  - 이동형 기업(약 30%): 인터넷뱅킹 건수가 줄고 같은 만큼 스마트뱅킹 건수가 늘어난다 (총 활동은 유지)
  - 위축형 기업(약 12%): 어느 달부터 모든 국내 채널 건수와 요구불입금이 크게 줄어든다
  - 외환 이용 기업(약 20%): 외환 건수는 다른 채널과 무관하게 움직인다
  - 36개월 완전관측 약 30%, 정확히 35개월 관측 약 18% (빈 달 위치: 첫 달 20%, 마지막 달 20%, 중간 60%)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cntlib import AMT_COLS, ATTR_COLS, CNT_COLS

BANDS = ["0건", "1건", "2건", "2건초과 5건이하", "5건초과 10건이하", "10건초과 20건이하",
         "20건초과 30건이하", "30건초과 40건이하", "40건초과 50건이하", "50건초과"]
EDGES = np.array([0.5, 1.5, 2.5, 5.5, 10.5, 20.5, 30.5, 40.5, 50.5])
SECTORS = ["제조업", "도매 및 소매업", "건설업", "부동산업", "운수 및 창고업", "정보통신업"]
FX_PROB = {"제조업": 0.35, "도매 및 소매업": 0.35}
SIDO, GRADE = ["대구", "경북", "서울", "부산", "경남"], ["최우수", "우수", "일반"]


def round_like_spec(x: np.ndarray) -> np.ndarray:
    unit = np.select([x < 1, x < 10, x < 100, x < 1000, x < 10000], [0.1, 1.0, 10.0, 100.0, 1000.0], default=10000.0)
    return np.round(x / unit) * unit


def make(n_firms: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(36)
    season = 1 + 0.2 * np.cos((t % 12 - 11) / 12 * 2 * np.pi)
    cols = {c: [] for c in ["법인ID", "기준년월"] + ATTR_COLS + AMT_COLS + CNT_COLS}
    nobs_list, migr, shrink = [], 0, 0
    shrink_ts = {}  # 법인ID -> 위축 시작월(0 기준), 테스트용 정답
    for i in range(n_firms):
        kind = rng.choice(["complete", "m35", "late", "early", "gap"], p=[0.30, 0.18, 0.20, 0.14, 0.18])
        obs = np.ones(36, dtype=bool)
        if kind == "m35":
            pos = rng.choice(["first", "last", "mid"], p=[0.2, 0.2, 0.6])
            obs[0 if pos == "first" else 35 if pos == "last" else rng.integers(1, 35)] = False
        elif kind == "late":
            obs[: rng.integers(1, 30)] = False
        elif kind == "early":
            obs[rng.integers(3, 33):] = False
        elif kind == "gap":
            for _ in range(rng.integers(1, 4)):
                s = rng.integers(2, 34)
                obs[s: s + rng.integers(1, 5)] = False
        if not obs.any():
            obs[rng.integers(0, 36)] = True
        if kind in ("late", "early", "gap") and obs.sum() >= 35:
            obs[rng.integers(1, 35)] = False  # 35·36개월 관측은 complete/m35 로만 만든다 (정답 개수를 정확히 알기 위해)
            if obs.sum() >= 35:
                obs[rng.integers(1, 35)] = False
        nobs_list.append(int(obs.sum()))

        sector = rng.choice(SECTORS)
        grade = rng.choice(GRADE, p=[0.2, 0.3, 0.5])
        duty = rng.choice(["Y", "N"], p=[0.3, 0.7])
        sido = rng.choice(SIDO)
        a = float(np.exp(rng.normal(1.6, 0.9)))
        prop = np.exp(rng.normal(0, 0.6, 5))
        base = a * np.array([0.6, 1.5, 0.8, 0.15, 0.7]) * prop  # 창구, 인터넷, 스마트, 폰, ATM
        lam = np.outer(base, season)  # (5,36)
        migrator = rng.random() < 0.30
        shrinker = (not migrator) and rng.random() < 0.17  # 전체 약 12%
        factor = np.ones(36)
        if migrator:
            migr += 1
            t0, L = rng.uniform(0, 12), rng.uniform(18, 30)
            p = np.clip((t - t0) / L, 0, 1)
            lam[1] = base[1] * 1.6 * season * (1 - 0.85 * p)
            lam[2] = base[1] * 1.6 * season * (0.1 + 0.9 * p)
        if shrinker:
            shrink += 1
            ts = rng.integers(14, 29)
            factor = np.where(t >= ts, rng.uniform(0.1, 0.3), 1.0)
            lam = lam * factor
        counts = np.stack([rng.poisson(lam[j]) for j in range(5)]).astype(float)  # (5,36)
        auto_l = np.exp(rng.normal(2.0, 0.6)) * (0.5 + 0.5 * factor)
        auto = rng.poisson(auto_l * season)
        fx_on = rng.random() < FX_PROB.get(sector, 0.10)  # 제조업·도매는 외환 이용이 많다 (전체 약 18%)
        if fx_on:
            fx_x = rng.poisson(rng.uniform(0.5, 4.0), 36)
            fx_m = rng.poisson(rng.uniform(0.5, 4.0), 36)
        else:
            fx_x = fx_m = np.zeros(36, dtype=int)
        all_counts = np.vstack([counts, auto[None, :], fx_x[None, :], fx_m[None, :]])  # (8,36)
        codes = np.searchsorted(EDGES, all_counts, side="left")

        tot = counts.sum(axis=0) + 1.0
        inflow = np.exp(rng.normal(3.5, 1.2)) * season * (tot / tot.mean()) ** 0.5 * np.exp(rng.normal(0, 0.3, 36)) * (0.3 + 0.7 * factor if shrinker else 1.0)
        outflow = inflow * np.exp(rng.normal(-0.05, 0.25, 36))
        balance = np.exp(rng.normal(3.5, 1.2)) * 1.5 * np.exp(rng.normal(0, 0.3, 36))
        fid = f"SYN_{i:06d}_{rng.integers(0, 16 ** 8):08x}"
        if shrinker:
            shrink_ts[fid] = int(ts)
        for m in np.flatnonzero(obs):
            cols["법인ID"].append(fid)
            cols["기준년월"].append(str((2023 + m // 12) * 100 + m % 12 + 1))
            cols["업종_대분류"].append(sector)
            cols["사업장_시도"].append(None if rng.random() < 0.03 else sido)
            cols["법인_고객등급"].append(grade)
            cols["전담고객여부"].append(duty)
            for c, v in zip(AMT_COLS, (inflow[m], outflow[m], balance[m])):
                cols[c].append(float(round_like_spec(np.array([v]))[0]))
            for j, c in enumerate(CNT_COLS):
                cols[c].append(BANDS[int(codes[j, m])])
    df = pd.DataFrame(cols)
    nobs = np.array(nobs_list)
    expected = {"rows": len(df), "firms": n_firms, "p36": int((nobs == 36).sum()), "p35": int((nobs == 35).sum()),
                "migrators": migr, "shrinkers": shrink, "shrink_ts": shrink_ts}
    return df, expected
