from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QALSParams:
    i_max: list[int]
    N_max: list[int]
    d_min: list[int]

KSP_EXAMPLES = [
    Path("QA4QUBO/ksp/ksp_1.txt"),
    Path("QA4QUBO/ksp/ksp_2.txt"),
    Path("QA4QUBO/ksp/ksp_3.txt"),
]

"""
    Test finale con 1000, 500, 350
"""
QALS_PARAMS = QALSParams(
    i_max=[10, 50, 100, 250, 500],
    N_max=[5,  25, 50,  125, 250],
    d_min=[4,  18, 35,  88,  175]
)

# k = quante volte risolvo il problema QUBO
MATCH_K_T = [
    (1000, 10),
    (2000, 10),
    (3000, 10),
    (4000, 10),
    (5000, 10)
]

LAMBDA_VALUE = "lambda_div_3"
# LAMBDA_VALUE = "lambda_650_dot_C"
# LAMBDA_VALUE = "lambda_div_C"