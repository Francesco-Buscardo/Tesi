# Quantum Annealing for Solving QUBO Problems

Python implementation of research: Pastorello, D. and Blanzieri, E., 2019. Quantum annealing learning search for solving QUBO problems. Quantum Information Processing, 18(10), p.303.

## Requirements

- Python 3.12 recommended
- `pip`
- Python virtual environment
- Dependencies listed in `requirements_ksp.txt`

## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Tesi/Quantum-Annealing-for-solving-QUBO-Problems
````

Replace `<repository-url>` with the actual repository URL.

### 2. Create a Virtual Environment

On Windows with Git Bash:

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

On Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements_ksp.txt

### 4. Run the Program

To start run:

```bash
python -m pip install -r requirements_ksp.txt
python ksp_start.py
```

## Project Structure

```text

```

## Notes

This project uses the D-Wave Ocean SDK.
For compatibility reasons, Python 3.12 is recommended.

If you are using Windows, Git Bash is recommended for running the commands shown above.

The Knapsack Problem input files are located in:

```text
QA4QUBO/ksp/
```
