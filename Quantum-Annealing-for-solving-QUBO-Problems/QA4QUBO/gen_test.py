import numpy as np
import csv
import re
import matplotlib.pyplot as plt

from pathlib import Path
from pathlib import Path
from os import path, makedirs

from QA4QUBO import app
import ksp_config as ksp_config


def remove_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def generate_folder_match_k_TIMES(file):
    if ksp_config.LAMBDA_VALUE == "lambda_div_3":
        lambda_folder = "lambda_div_3"
    elif ksp_config.LAMBDA_VALUE == "lambda_650_dot_C":
        lambda_folder = "lambda_650_dot_C"
    elif ksp_config.LAMBDA_VALUE == "lambda_6500_dot_C":
        lambda_folder = "lambda_6500_dot_C"
    elif ksp_config.LAMBDA_VALUE == "lambda_div_C":
        lambda_folder = "lambda_div_C"
    else:
        lambda_folder = ""

    ksp_name = Path(file).stem
    folder   = f"{ksp_config.TEST_FOLDER}/{ksp_name}/{lambda_folder}/"

    makedirs(folder, exist_ok=True)

    for k, t in ksp_config.MATCH_K_T:
        filename = f"file_{k}_{t}.txt"
        filepath = path.join(folder, filename)

        with open(filepath, "w") as f:
            f.write(f"k     = {k}\n")
            f.write(f"TIMES = {t}\n")

        print(f"Create: {filepath}")
    
    return folder

def run_match_k_TIMES(file, n, capacity, items, _Q):
    folder = generate_folder_match_k_TIMES(file)

    for (k, TIMES) in ksp_config.MATCH_K_T:
        filepath = path.join(folder, f"file_{k}_{TIMES}.txt")

        with open(filepath, "a") as f:
            f.write("\nQALS\n")
            f.write(remove_ansi(app.app1(TIMES, k, _Q, n, capacity, items)))

            f.write("\nNO QALS\n\n")
            f.write(remove_ansi(app.app2(TIMES, k, _Q, n, capacity, items)))

def vector_to_string(v):
    return "".join(str(int(x)) for x in v)

def save_single_hamming_plot(matrix, title, y_label, plot_path):
    ham_distances = [item[2] for item in matrix]
    t_values = list(range(len(ham_distances)))

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, ham_distances)

    plt.xlabel("t")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def write_single_matrix_to_csv(writer, label, i_max, n_max, d_min, matrix):
    writer.writerow([])
    writer.writerow([label])
    writer.writerow([i_max, n_max, d_min])

    for item in matrix:
        z        = item[0]
        fQ       = item[1]
        hamm_d   = item[2]
        hamm_vec = item[3]

        changed_items = np.where(hamm_vec == 1)[0].tolist()

        writer.writerow([
            vector_to_string(z),
            fQ,
            hamm_d,
            changed_items
        ])

def save_diagnostics_to_csv(k, i_max, n_max, d_min, solutions_matrix_star, solutions_matrix_best, solutions_matrix_opt):
    folder = Path(f"test/Diagnostica_QALS/{ksp_config.KSP_FILE}/{ksp_config.LAMBDA_VALUE}/{k}")
    folder.mkdir(parents=True, exist_ok=True)
    
    """path_csv = folder / "diagnostics.csv"
    file_exists = path_csv.exists()
    
    with open(path_csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "i_max",
                "N_max",
                "d_min",
                "z",
                "fQ",
                "hamming_distance",
                "changed_items"
            ])

        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_star",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_star
        )
        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_best",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_best
        )
        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_opt",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_opt
        )"""
    
    save_single_hamming_plot(
        matrix=solutions_matrix_star,
        title=r"Andamento $d_H(z_t, z_{star})$",
        y_label="Hamming distance",
        plot_path=f"{folder}/diagnostics_Hamming_star_i{i_max}_N{n_max}_d{d_min}.png"
    )
    save_single_hamming_plot(
        matrix=solutions_matrix_best,
        title=r"Andamento $d_H(z_t, z_{best})$",
        y_label="Hamming distance",
        plot_path=f"{folder}/diagnostics_Hamming_best_i{i_max}_N{n_max}_d{d_min}.png"
    )
    save_single_hamming_plot(
        matrix=solutions_matrix_opt,
        title=r"Andamento $d_H(z_t, z_{opt})$",
        y_label="Hamming distance",
        plot_path=f"{folder}/diagnostics_Hamming_opt_i{i_max}_N{n_max}_d{d_min}.png"
    )
    