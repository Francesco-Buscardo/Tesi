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

### 3. Install the Required Packages

First, upgrade `pip`, `setuptools`, and `wheel`:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Then install the project dependencies:

```bash
python -m pip install -r requirements_ksp.txt
```

### 4. Run the Program

To start the Knapsack Problem experiment, run:

```bash
python ksp_start.py
```

## Project Structure

```text
Tesi/
├── Docs/
│   ├── Alg_1_genreal_idea.md
│   ├── Alg_2_QUBO_idea.md
│   ├── QALS_implementations.pdf
│   ├── Quantum_Annealing_for_QUBO_problems.pdf
│   ├── dataset_generator.c
│   ├── generator
│   └── test.in
│
├── Quantum-Annealing-for-solving-QUBO-Problems/
│   ├── QA4QUBO/
│   │   ├── ksp/
│   │   │   ├── README.txt
│   │   │   ├── ksp_1.txt
│   │   │   ├── ksp_2.txt
│   │   │   └── ksp_3.txt
│   │   │
│   │   ├── qap/
│   │   │   ├── Bur26.txt
│   │   │   ├── Chr12.txt
│   │   │   ├── Esc16.txt
│   │   │   ├── Esc64.txt
│   │   │   ├── Lipa70.txt
│   │   │   └── Nug28.txt
│   │   │
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── colors.py
│   │   ├── ksp.py
│   │   ├── matrix.py
│   │   ├── mksp.py
│   │   ├── script.py
│   │   ├── solver.py
│   │   ├── test.py
│   │   ├── tsp.py
│   │   └── vector.py
│   │
│   ├── test/
│   │   ├── scale_QUBO/
│   │   ├── test_i_max/
│   │   ├── test_match_k_TIMES/
│   │   └── test_match_k_TIMES_fisso/
│   │
│   ├── LICENSE
│   ├── README.md
│   ├── ksp_config.py
│   ├── ksp_start.py
│   ├── requirements.txt
│   ├── requirements_ksp.txt
│   └── start.py
│
├── tex/
│   ├── ORDINE.txt
│   ├── Obb.pdf
│   ├── Report2.pdf
│   ├── Report_5_i_max.pdf
│   ├── Report_6.pdf
│   ├── Sviluppo_Classical_ksp.pdf
│   ├── Sviluppo_mksp.pdf
│   ├── Swap_k_TMES.pdf
│   └── Tesi.pdf
│
├── .gitignore
└── ToDo.md
```

## Notes

This project uses the D-Wave Ocean SDK.
For compatibility reasons, Python 3.12 is recommended.

If you are using Windows, Git Bash is recommended for running the commands shown above.

The Knapsack Problem input files are located in:

```text
QA4QUBO/ksp/
```
