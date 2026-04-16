import os
import io
import re
from contextlib import redirect_stdout

from pymhlib.demos.mkp import MKPInstance, MKPSolution
from pymhlib.demos.common import run_optimization

def main():
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        run_optimization("MKP", MKPInstance, MKPSolution, "file.txt")

    output = buffer.getvalue()

    best_solution = None
    best_obj = None

    m1 = re.search(r"T best solution:\s*\[([^\]]*)\]", output)
    if m1:
        text = m1.group(1).strip()
        best_solution = [] if not text else [int(x) for x in text.split()]

    m2 = re.search(r"T best obj:\s*([0-9.+-]+)", output)
    if m2:
        best_obj = float(m2.group(1))

    print("Best solution:", best_solution)
    print("Best objective value:", best_obj)

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    main()