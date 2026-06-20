from dataclasses import dataclass
from pathlib import Path

@dataclass
class QALSParams:
    i_max: list[int]
    N_max: list[int]
    d_min: list[int]

TEST_FOLDER  = Path("test/Diagnostica_QALS/")

KSP_EXAMPLES = [
    # Path("QA4QUBO/ksp/ksp_1.txt")
    # Path("QA4QUBO/ksp/ksp_2.txt")
    Path("QA4QUBO/ksp/ksp_3.txt")
]

KSP_FILE = "ksp_3"

QALS_PARAMS = QALSParams(       
    i_max=[10, 50, 100, 250, 500, 1000],
    N_max=[5,  25, 50,  125, 250, 500],
    d_min=[4,  18, 35,  88,  175, 350]
)

# k = quante volte risolvo il problema QUBO
TIMES = 1
MATCH_K_T = [
   (1000, TIMES),
   (1250, TIMES),
   (1500, TIMES),
   (1750, TIMES),
   (2000, TIMES)
]

# 1, 10, 20
SCALE_QUBO = 1

# LAMBDA_VALUE = "lambda_div_3"
LAMBDA_VALUE = "lambda_650_dot_C"
# LAMBDA_VALUE = "lambda_6500_dot_C"
# LAMBDA_VALUE = "lambda_div_C"

DECAY_FACTOR = 1