import os
import re
import ast
import matplotlib.pyplot as plt


INPUT_DIR  = "match_k_TIMES"
OUTPUT_DIR = "plots"
METRICS    = [
    "Avg Profit GAP",
    "Avg Weight GAP",
    "Avg fQ",
    "fQ Min Found",
    "Profit Found",
    "Weight Found",
    "Profit GAP",
    "Weight GAP",
    "n of fQ Min Found",
]

def clean_metric_name(name):
    return name.replace(" ", "_").replace("fQ", "fq").lower()

def parse_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    k_match = re.search(r"k\s*=\s*(\d+)", text)
    times_match = re.search(r"TIMES\s*=\s*(\d+)", text)

    if not k_match or not times_match:
        return None

    k = int(k_match.group(1))
    times = int(times_match.group(1))

    data = {
        "k": k,
        "TIMES": times,
        "QALS": {},
        "NO QALS": {},
    }

    sections = re.split(r"\n\s*(QALS|NO QALS)\s*\n", text)

    current_section = None

    for part in sections:
        part = part.strip()

        if part in ["QALS", "NO QALS"]:
            current_section = part
            continue

        if current_section is None:
            continue

        for line in part.splitlines():
            match = re.match(r"\[(.*?)\]\s*(.*)", line.strip())

            if not match:
                continue

            key = match.group(1).strip()
            value = match.group(2).strip()

            if key == "Items":
                try:
                    data[current_section][key] = ast.literal_eval(value)
                except Exception:
                    data[current_section][key] = None
            else:
                try:
                    data[current_section][key] = float(value)
                except ValueError:
                    pass

    return data

def load_all_data():
    results = []

    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(INPUT_DIR, filename)
        parsed = parse_file(path)

        if parsed is not None:
            results.append(parsed)

    results.sort(key=lambda x: (x["k"], x["TIMES"]))
    return results

def plot_metric(results, metric, section):
    grouped = {}

    for row in results:
        k = row["k"]
        times = row["TIMES"]

        if metric not in row[section]:
            continue

        grouped.setdefault(k, []).append((times, row[section][metric]))

    if not grouped:
        return

    plt.figure()

    for k, values in sorted(grouped.items()):
        values.sort()
        x = [v[0] for v in values]
        y = [v[1] for v in values]

        plt.plot(x, y, marker="o", label=f"k={k}")

    plt.xlabel("TIMES")
    plt.ylabel(metric)
    plt.title(f"{metric} - {section}")
    plt.xscale("log")
    plt.grid(True)
    plt.legend()

    filename = f"{section.lower().replace(' ', '_')}_{clean_metric_name(metric)}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()

def plot_items(results, section):
    rows = []

    for row in results:
        items = row[section].get("Items")

        if not items:
            continue

        rows.append((row["k"], row["TIMES"], items))

    if not rows:
        return

    for k, times, items in rows:
        plt.figure()
        plt.bar(range(len(items)), items)
        plt.xlabel("Item index")
        plt.ylabel("Selected")
        plt.title(f"Items - {section} - k={k}, TIMES={times}")
        plt.xticks(range(len(items)))
        plt.ylim(0, 1.2)
        plt.grid(axis="y")

        filename = f"{section.lower().replace(' ', '_')}_items_k{k}_times{times}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches="tight")
        plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_all_data()

    if not results:
        print(f"Nessun file .txt trovato in '{INPUT_DIR}'")
        return

    for section in ["QALS", "NO QALS"]:
        for metric in METRICS:
            plot_metric(results, metric, section)

        plot_items(results, section)

    print(f"Grafici salvati nella cartella '{OUTPUT_DIR}'")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')

    main()