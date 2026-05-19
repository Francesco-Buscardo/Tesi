from dataclasses import dataclass
from pathlib import Path


@dataclass
class QALSParams:
    i_max: list[int]
    N_max: list[int]
    d_min: list[int]

TEST_FOLDER = Path("test/scale_QUBO/20/")

KSP_EXAMPLES = [
    # Path("QA4QUBO/ksp/ksp_1.txt"),
    Path("QA4QUBO/ksp/ksp_2.txt"),
    Path("QA4QUBO/ksp/ksp_3.txt"),
]

QALS_PARAMS = QALSParams(
    i_max=[10, 50, 100, 250, 500, 1000],
    N_max=[5,  25, 50,  125, 250, 500],
    d_min=[4,  18, 35,  88,  175, 350]
)

# k = quante volte risolvo il problema QUBO
MATCH_K_T = [
    (1000, 5),
    (1250, 5),
    (1500, 5),
    (1750, 5),
    (2000, 5)
]

SCALE_QUBO = 1
# SCALE_QUBO = 10
# SCALE_QUBO = 20

LAMBDA_VALUE = "lambda_div_3"
# LAMBDA_VALUE = "lambda_650_dot_C"
# LAMBDA_VALUE = "lambda_6500_dot_C"
# LAMBDA_VALUE = "lambda_div_C"