"""cntlib 의 핵심 함수 단위 점검 (데이터 불필요).  python test_cntlib.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import cntlib as cl  # noqa: E402

ok_all = True


def check(name, cond):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# 쌍 맞춤
M = np.full((2, 36), np.nan); M[0, :] = 5.0; M[0, 20] = np.nan; M[1, :] = 2.0; M[0, 30] = 9.0
rc, pr = cl.paired_means(M, 30, 3, 2)
check("paired_means: 빈 달이 없는 창은 단순 평균", abs(rc[0] - (5 + 5 + 9) / 3) < 1e-12 and abs(pr[0] - 5.0) < 1e-12)
M2 = M.copy(); M2[0, 16] = np.nan
rc, pr = cl.paired_means(M2, 30, 3, 2)
check("paired_means: 1년 전 창의 빈 달에 대응하는 최근 창의 달도 함께 뺀다", abs(rc[0] - (5 + 9) / 2) < 1e-12 and abs(pr[0] - 5.0) < 1e-12)
check("paired_means: 짝지은 달이 need 미만이면 판정 불가(NaN)", np.isnan(cl.paired_means(M2, 30, 3, 3)[0][0]))
check("min_pairs: k=1 이어도 최소 1쌍, k=3 이면 2쌍, k=6 이면 5쌍", (cl.min_pairs(1), cl.min_pairs(3), cl.min_pairs(6)) == (1, 2, 5))

# 구간 하한/상한
codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]])
lo, hi = cl.bound_count(codes, "low"), cl.bound_count(codes, "high", 100)
check("bound_count: 하한/상한 매핑", lo[0, :10].tolist() == [0, 1, 2, 3, 6, 11, 21, 31, 41, 51]
      and hi[0, :10].tolist() == [0, 1, 2, 5, 10, 20, 30, 40, 50, 100] and np.isnan(lo[0, 10]))

# 인과적 채움
row = np.array([[2, np.nan, 4, np.nan, 6]], dtype=float)
check("fill_causal: s=4 이면 양쪽 관측으로 보간", cl.fill_causal(row, 4)[0].tolist() == [2, 3, 4, 5, 6])
r2 = cl.fill_causal(row, 2)[0]
check("fill_causal: s=2 이면 s 이후 열은 NaN, 과거 빈 달은 s 까지의 값으로 보간", r2[:3].tolist() == [2, 3, 4] and np.isnan(r2[3:]).all())
check("fill_causal: s 자체가 미관측이면 판정 불가(행 전체 NaN)", np.isnan(cl.fill_causal(row, 3)).all())
a = np.array([[2, np.nan, 4, np.nan, 6]], dtype=float)
b = np.array([[2, np.nan, 4, np.nan, 99]], dtype=float)
check("fill_causal: s 이후 값이 달라져도 s 시점 결과는 같다 (미래 정보 누수 없음)",
      np.array_equal(cl.fill_causal(a, 2), cl.fill_causal(b, 2), equal_nan=True))
lead = np.array([[np.nan, np.nan, 3, 3, 5]], dtype=float)
check("fill_causal: 첫 관측 이전은 가장 가까운 이후 관측(<= s)으로", cl.fill_causal(lead, 4)[0].tolist() == [3, 3, 3, 3, 5])

print("\n결과:", "모두 통과" if ok_all else "실패 있음")
sys.exit(0 if ok_all else 1)
