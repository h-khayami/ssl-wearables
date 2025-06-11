import subprocess
import csv
import os
import re
from datetime import datetime

# ------------------------- Configuration -------------------------
k_shot_values = [5, 20, 50]
lr = 0.0001
freeze_weights = False

experiments = [
    {
        "name": "MLP_OPT-Personalized",
        "flip_path": "/data/ssl_wearable/model/2025-05-02_09:40:41tmp.pt",
        "split": "k_shot_fixed",
        "load_weights": True,
        "load_personalized": True,
        "plot_legend": "OPT -> Personalized",
        "conditions": f"retrained on mymove loso and then personalized with fixed test set lr={lr}",
        "model": "MLP"
    },
    {
        "name": "MLP_Broader-OPT",
        "flip_path": "/data/ssl_wearable/model/2025-04-14_22:45:26tmp.pt",
        "split": "held_one_subject_out_k_shot",
        "load_weights": True,
        "load_personalized": False,
        "plot_legend": "Broader -> OPT",
        "conditions": f"pretrained on realworld_wisdm and then finetuned on LOSO with fixed test set lr={lr}",
        "model": "MLP"
    },
    {
        "name": "MLP Broader-Personalized",
        "flip_path": "/data/ssl_wearable/model/2025-04-14_22:45:26tmp.pt",
        "split": "k_shot_fixed",
        "load_weights": True,
        "load_personalized": False,
        "plot_legend": "Broader -> Personalized",
        "conditions": f"pretrained on realworld_wisdm and then personalized with fixed test set lr={lr}",
        "model": "MLP"
    },
    {
        "name": "MLP_Broader-OPT-Personalized",
        "flip_path": "/data/ssl_wearable/model/2025-04-15_13:15:18tmp.pt",
        "split": "k_shot_fixed",
        "load_weights": True,
        "load_personalized": True,
        "plot_legend": "Broader -> OPT -> Personalized",
        "conditions": f"pretrained on realworld_wisdm and finetuned on mymove loso and then personalized with fixed test set lr={lr}",
        "model": "MLP"
    },
    {
        "name": "MLP_Personalized",
        "flip_path": "",
        "split": "k_shot_fixed",
        "load_weights": False,
        "load_personalized": False,
        "plot_legend": "Personalized (no pretraining)",
        "conditions": "personalized directly with fixed test set lr={lr}",
        "model": "MLP"
    }
]
# ---------------------------------------------------------------
# Create timestamped log directory
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
report_folder = f"data/reports/"
log_folder = f"data/logs/"
os.makedirs(report_folder, exist_ok=True)

# Prepare CSV file
csv_file = os.path.join(report_folder, "experiment_results.csv")
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a" if file_exists else "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=[
        "path",
        "validation", "split", "plot_legend", "conditions", "comments",
        "freeze_weight", "model"
    ])
    if not file_exists:
        writer.writeheader()
    writer.writeheader()

    for exp in experiments:
        for k in k_shot_values:
            # Set up log file name
            log_file = f"{exp['name'].replace(' ', '_')}K{k}_LR{lr}.log"
            log_path = os.path.join(log_folder, log_file)
            print(f"Log path: {log_path}")
            # Check if already completed
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log_text = f.read()
                    match = re.search(r'Classification report saved to.*?/reports/([^/]+)/', log_text)
                    if match:
                        report_folder = match.group(1)
                        print(f"✅ Skipping completed: {log_file}")
                        writer.writerow({
                            "path": report_folder,
                            "validation": "personalization" if exp["split"]=="k_shot_fixed" else "cross-user",
                            "split": f"{k}-shot" if exp["split"]=="k_shot_fixed" else f"LOSO {k}-shot",
                            "plot_legend": exp["plot_legend"],
                            "conditions": exp["conditions"],
                            "comments": "",
                            "freeze_weight": freeze_weights,
                            "model": exp["model"]
                        })
                        continue
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            # Build command
            cmd = [
                "python", "downstream_task_evaluation.py",
                f"k_shot={k}",
                f"split_method={exp['split']}",
                f"evaluation.learning_rate={lr}",
                f"evaluation.load_weights={str(exp['load_weights'])}",
                f"evaluation.load_personalized_weights={str(exp['load_personalized'])}",
            ]
            if exp["flip_path"]:
                cmd.append(f"evaluation.flip_net_path={exp['flip_path']}")

            print(f"🚀 Running: {exp['name']} | k={k}")
            with open(log_path, "w") as logfile:
                subprocess.run(cmd, stdout=logfile, stderr=logfile)

            # Check result after run
            with open(log_path) as f:
                finished = "Classification report saved to" in f.read()

            writer.writerow({
                "path": timestamp,
                "validation": "personalization" if exp["split"]=="k_shot_fixed" else "cross-user",
                "split": f"{k}-shot" if exp["split"]=="k_shot_fixed" else f"LOSO {k}-shot",
                "plot_legend": exp["plot_legend"],
                "conditions": exp["conditions"],
                "comments": "FAILED" if not finished else "",
                "freeze_weight": freeze_weights,
                "model": exp["model"]
            })