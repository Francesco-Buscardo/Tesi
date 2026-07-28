from dataclasses import dataclass
from pathlib import Path

@dataclass
class QALSParams:
    i_max: list[int]
    N_max: list[int]
    d_min: list[int]

TEST_FOLDER  = Path("test/Lambda_dinamico/")

KSP_EXAMPLES = [
    #Path("QA4QUBO/ksp/ksp_1.txt")
    #Path("QA4QUBO/ksp/ksp_2.txt")
    Path("QA4QUBO/ksp/ksp_3.txt")
]

# i_max=[10, 50, 100, 250, 500, 1000]
# N_max=[5,  25, 50,  125, 250, 500 ]
# d_min=[4,  18, 35,  88,  175, 350 ]
QALS_PARAMS = QALSParams(       
    i_max=[250],
    N_max=[125],
    d_min=[88]
)

# k = quante volte risolvo il problema QUBO
TIMES = 1
MATCH_K_T = [
   (2000, TIMES)
]

SCALE_QUBO = 1

LAMBDA_VALUE = "lambda_div_3"
# LAMBDA_VALUE = "lambda_650_dot_C"
# LAMBDA_VALUE = "lambda_6500_dot_C"
# LAMBDA_VALUE = "lambda_div_C"

DECAY_FACTOR = 1

# PARAMS DYNAMIC LAMBDA
R_MIN = 0.05
R_MAX = 0.8
PAR_M = 5

P_STG_COUNTER        = 0.1
NUM_S_STAGNATIONS    = 3
AMPLIFICATION_FACTOR = 50

MOVING_AVG_PERIOD = 10
LAM_ABS_MAX_MULT  = 10

BETA = 5