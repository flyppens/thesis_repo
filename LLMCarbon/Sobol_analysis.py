
#python Sobol_analysis.py -c config_all.json --sensitivity --row 10 --samples 1000

import argparse
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import LLM
from embodied import hardware_list


try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from joblib import Parallel, delayed
except ImportError:
    saltelli = sobol = Parallel = delayed = None



def predict_train_co2eq(model_row: pd.Series, noise: np.ndarray) -> float:
    params_b = model_row["parameter # (B)"] * noise[0]
    tokens_b = model_row["token # (B)"] * noise[1]
    hw_eff   = model_row["hardware efficiency"] * noise[2]

    model = LLM(
        model_row["name"],
        model_row["type"],
        params_b,
        model_row["base model param. # (B)"],
        tokens_b,
        model_row["CO2eq/KWh"],
        model_row["PUE"],
        model_row["computing device"],
        model_row["device TPD (W)"],
        model_row["avg. system power (W)"],
        model_row["peak TFLOPs/s"],
        model_row["achieved TFLOPs/s"],
        hw_eff,
        model_row["device #"],
        model_row["total zettaFLOPs"],
        model_row["training days"],
        model_row["actual tCO2eq"],
    )
    model.training_co2eq()
    #return model.predicted_co2eq_train
    return model.train_delta



def run_sensitivity(model_row: pd.Series, base_samples: int = 1000) -> None:
    if saltelli is None:
        raise ImportError("SALib or joblib missing — install extras to use --sensitivity")

    problem = {
        "num_vars": 3,
        "names": ["params", "tokens", "eff"],
        "bounds": [
            [0.94,  1.0279],   
            [0.98,  1.02],  
            [0.16,  3],
        ],
    }

    print("\nRunning Sobol sensitivity analysis on '{}'. This may take a minute...".format(model_row["name"]))
    X = saltelli.sample(problem, base_samples, calc_second_order=True)

    # Vectorised evaluation in parallel
    Y = Parallel(n_jobs=-1)(
        delayed(predict_train_co2eq)(model_row, x) for x in X
    )
    Y = np.asarray(Y)

    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)

    header = f"{'Feature':<15} {'S1':>6} {'ST':>6}"
    print("\n" + header)
    print("-" * len(header))
    for name, s1, st in zip(problem["names"], Si["S1"], Si["ST"]):
        print(f"{name:<15} {s1:6.3f} {st:6.3f}")
    
    print("\nSecond-order (interaction) indices")
    header2 = f"{'Feature pair':<25} {'S2':>6}"
    print(header2)
    print("-" * len(header2))
    names = problem["names"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"{names[i]:<8} & {names[j]:<15} {Si['S2'][i, j]:6.3f}")  

 
    try:
        plt.figure(figsize=(6, 3.5))
        plt.barh(problem["names"], Si["ST"])
        plt.xlabel("Total Sobol index ($S_{Ti}$)")
        plt.title(f"Training‑CO₂ drivers — {model_row['name']}")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMCarbon validation + sensitivity tutorial")
    parser.add_argument("-c", "--config", type=str, default="config_all.json", help="JSON config file")
    parser.add_argument("--sensitivity", action="store_true", help="Run global sensitivity analysis")
    parser.add_argument("--row", type=int, default=0, help="Row index in database for sensitivity study")
    parser.add_argument("--samples", type=int, default=1000, help="Base sample size for Saltelli")

    
    parser.add_argument("--database", type=str, default="database.csv")
    parser.add_argument("--hardware", type=str, default="hardware.csv")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        parser.set_defaults(**cfg)            
        args = parser.parse_args()            

    models_df = pd.read_csv(args.database)
    print("Operational carbon‑footprint validation\n")

    for idx in range(5, len(models_df.index)):
        r = models_df.iloc[idx]
        model = LLM(
            r["name"],
            r["type"],
            r["parameter # (B)"],
            r["base model param. # (B)"],
            r["token # (B)"],
            r["CO2eq/KWh"],
            r["PUE"],
            r["computing device"],
            r["device TPD (W)"],
            r["avg. system power (W)"],
            r["peak TFLOPs/s"],
            r["achieved TFLOPs/s"],
            r["hardware efficiency"],
            r["device #"],
            r["total zettaFLOPs"],
            r["training days"],
            r["actual tCO2eq"],
        )
        model.training_co2eq()
        model.inference_co2eq()
        print(f"{model.name}\t{model.train_delta:.3f}")

    hardware_df = pd.read_csv(args.hardware)
    hard_list = hardware_list(hardware_df)
    print("\nEmbodied carbon‑footprint validation")
    hard_list.embodied_co2eq(0.4889, 0.102)   # intensity & recycling factor from paper
    print(f"Total embodied CO₂eq (t): {hard_list.total_embodied_co2eq:,.1f}")

    if args.sensitivity:
        if args.row < 0 or args.row >= len(models_df):
            raise IndexError("--row index out of range for database.csv")
        run_sensitivity(models_df.iloc[args.row], base_samples=args.samples)


if __name__ == "__main__":
    main()
