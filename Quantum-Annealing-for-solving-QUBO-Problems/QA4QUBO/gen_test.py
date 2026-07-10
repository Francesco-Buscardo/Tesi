import re
import matplotlib.pyplot as plt

from pathlib import Path
from os import path, makedirs

import ksp_config as ksp_config


def remove_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def get_lambda_folder():
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
    return lambda_folder

def get_folder_path(file):
    lambda_folder = get_lambda_folder()

    ksp_name = Path(file).stem
    folder   = f"{ksp_config.TEST_FOLDER}/{ksp_name}/{lambda_folder}/"
    makedirs(folder, exist_ok=True)

    return folder

def generate_folder_match_k_TIMES(file):
    folder = get_folder_path(file)

    for k, t in ksp_config.MATCH_K_T:
        filename = f"file_{k}_{t}.txt"
        filepath = path.join(folder, filename)

        with open(filepath, "w") as f:
            f.write(f"k     = {k}\n")
            f.write(f"TIMES = {t}\n")

        print(f"Create: {filepath}")
    
    return folder

def vector_to_string(v):
    return "".join(str(int(x)) for x in v)

def moving_average(values, window=10):
    smoothed = []

    for i in range(len(values)):
        start = max(0, i-window+1)
        window_values = values[start:i+1]
        smoothed.append(sum(window_values) / len(window_values))

    return smoothed

def gen_plot(t_values, val, title, y_label, smooth, window, plot_path):
    plt.figure(figsize=(10, 5))

    plt.plot(
        t_values,
        val,
        alpha=0.35,
        label=y_label
    )
    plt.plot(
        t_values,
        smooth,
        linewidth=1,
        linestyle="--",
        label=f"Moving average (window={window})"
    )

    plt.xlabel("t")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_plot_solution_matrix(matrix, title_dh, title_fq, folder):
    plot_path_dh = Path(folder) / title_dh
    plot_path_fq = Path(folder) / title_fq

    fQ            = [item[1] for item in matrix]
    ham_distances = [item[2] for item in matrix]

    window        = max(5, len(ham_distances) // 20)
    t_values      = list(range(len(ham_distances)))

    ham_distances_smooth = moving_average(ham_distances, window=window)
    fQ_smooth = moving_average(fQ, window=window)

    # plot DH
    gen_plot(t_values, ham_distances, title_dh, "DH", ham_distances_smooth, window, plot_path_dh)
    # plot fQ
    gen_plot(t_values, fQ, title_fq, "fQ", fQ_smooth, window, plot_path_fq)