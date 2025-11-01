import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import config as cfg

# -------------------------
# Config & ordering
# -------------------------
ALPHA = 0.05
MENSA = "mensa"

dataset_names = ['mimic_me', 'rotterdam_me', 'proact_me', 'ebmt_me']

# Exact display order (and labels) you asked for
disc_metrics_display = ["GlobalCI", "LocalCI", "AUC"]
calib_metrics_display = ["IBS", "MAE", "D-calib"]

# Map display name -> column name in CSV
METRIC_COL = {
    "GlobalCI": "GlobalCI",
    "LocalCI": "LocalCI",
    "AUC": "AUC",
    "IBS": "IBS",
    "MAE": "MAEM",     # CSV uses MAEM; we show "MAE"
    "D-calib": "DCalib"
}

# Orientation: higher is better for these
HIGHER_BETTER = {"GlobalCI", "LocalCI", "AUC", "D-calib"}  # D-calib treated as higher=better (larger p-values = better)

# Datasets where your earlier code scaled MAE by 1/100
MAE_DIV100_DATASETS = {"mimic_me", "rotterdam_me", "ebmt_me"}

# Minimum effect size of interest (absolute points; your CI/AUC/IBS are already *100)
MESI = {
    "GlobalCI": 1.0,
    "LocalCI": 1.0,
    "AUC": 1.0,
    "IBS": 1.0,
    "MAE": 1.0,     # remember this is MAEM internally; you already divide by 100 for some datasets
}

# -------------------------
# Data loading & prep
# -------------------------
def load_results():
    path = Path(cfg.RESULTS_DIR) / "multi_event.csv"
    df = pd.read_csv(path)

    # Match your previous scaling (constant across models within a dataset)
    for col in ["GlobalCI", "LocalCI", "AUC", "IBS"]:
        if col in df.columns:
            df[col] = df[col] * 100.0

    # Average over events -> one row per (ModelName, DatasetName, Seed)
    df_avg = (
        df.groupby(["ModelName", "DatasetName", "Seed"], as_index=False)
          .mean(numeric_only=True)
    )
    return df_avg

# -------------------------
# Helpers
# -------------------------
def get_metric_series(df, dataset, model, metric_display):
    """Return per-seed series for a given dataset/model/metric (after needed scaling)."""
    col = METRIC_COL[metric_display]
    sub = df[(df["DatasetName"] == dataset) & (df["ModelName"] == model)][["Seed", col]].dropna()
    s = sub.set_index("Seed")[col].copy()

    # Replicate your MAE per-dataset scaling
    if metric_display == "MAE" and dataset in MAE_DIV100_DATASETS:
        s = s / 100.0

    return s

def get_aligned_pairs(df, dataset, metric_display, baseline):
    mensa_s = get_metric_series(df, dataset, MENSA, metric_display)
    base_s  = get_metric_series(df, dataset, baseline, metric_display)
    merged = pd.concat([mensa_s.rename("mensa"), base_s.rename("base")], axis=1, join="inner").dropna()
    x = merged["mensa"].to_numpy(float)
    y = merged["base"].to_numpy(float)
    return x, y

def paired_ttest(x, y):
    if len(x) < 2 or len(y) < 2:
        return np.nan
    stat, p = ttest_rel(x, y, nan_policy="omit")
    return float(p)

def is_worse(metric_display, mensa_mean, base_mean, significant):
    if not significant:
        return False
    if metric_display in HIGHER_BETTER:
        return base_mean < mensa_mean
    return base_mean > mensa_mean  # lower-better (IBS, MAE)

# -------------------------
# Analysis
# -------------------------
def analyze(df):
    baselines = sorted([m for m in df["ModelName"].unique() if m != MENSA])
    rows = []

    ordered_metrics = disc_metrics_display + calib_metrics_display

    for dataset in dataset_names:
        for metric_display in ordered_metrics:
            col = METRIC_COL[metric_display]
            if col not in df.columns:
                continue

            pvals, models, mensa_means, base_means, n_pairs_list = [], [], [], [], []

            for model in baselines:
                x, y = get_aligned_pairs(df, dataset, metric_display, model)
                n_pairs = min(len(x), len(y))
                p = paired_ttest(x, y)

                pvals.append(p)
                models.append(model)
                mensa_means.append(np.mean(x) if len(x) else np.nan)
                base_means.append(np.mean(y) if len(y) else np.nan)
                n_pairs_list.append(n_pairs)

            pvals_arr = np.array(pvals, dtype=float)
            valid = ~np.isnan(pvals_arr)

            reject = np.zeros_like(pvals_arr, dtype=bool)
            p_adj = np.full_like(pvals_arr, np.nan, dtype=float)

            if valid.any():
                # More conservative adjustment if desired:
                # method can be "holm", "hommel", or "bonferroni"
                reject_valid, p_adj_valid, _, _ = multipletests(
                    pvals_arr[valid], alpha=ALPHA, method="holm"
                )
                reject[valid] = reject_valid
                p_adj[valid] = p_adj_valid

            # ---- Practical significance setup ----
            def passes_practical(metric_display, mensa_mean, base_mean):
                if metric_display == "D-calib":
                    return False  # do not dagger based on D-calib numeric differences
                delta = (mensa_mean - base_mean) if metric_display in {"GlobalCI","LocalCI","AUC"} else (base_mean - mensa_mean)
                return (not np.isnan(delta)) and (delta >= MESI.get(metric_display, 0.0))
            # --------------------------------------

            for model, p_raw, p_corr, sig, mm, bm, npr in zip(
                models, pvals_arr, p_adj, reject, mensa_means, base_means, n_pairs_list
            ):
                # after you have mm (MENSA_mean), bm (Baseline_mean), and sig (Holm/Hommel/Bonferroni)
                statistically_worse = bool(sig) and (
                    (metric_display in {"GlobalCI","LocalCI","AUC"} and bm < mm) or
                    (metric_display in {"IBS","MAE"} and bm > mm)
                )
                worse = statistically_worse and passes_practical(metric_display, mm, bm)

                rows.append({
                    "Dataset": dataset,
                    "Metric": metric_display,
                    "Model": model,
                    "n_pairs": int(npr),
                    "p_uncorrected": p_raw,
                    "p_holm": p_corr,
                    "significant": bool(sig),
                    "worse_than_mensa": worse,
                    "MENSA_mean": mm,
                    "Baseline_mean": bm,
                })

    return pd.DataFrame(rows)


# -------------------------
# Pretty printing (ordered)
# -------------------------
def print_results(res_df):
    # Discrimination section in fixed metric order
    print("\n=== Discrimination Metrics († = significantly worse than MENSA, Holm α=0.05) ===\n")
    for dataset in dataset_names:
        for metric in disc_metrics_display:
            sub = res_df[(res_df["Dataset"] == dataset) & (res_df["Metric"] == metric)]
            if sub.empty:
                continue
            worse_models = sub.loc[sub["worse_than_mensa"], "Model"].tolist()
            tag = " †" if worse_models else ""
            print(f"{dataset:<12} | {metric:<9} | Worse: {', '.join(worse_models)}{tag}")

    # Calibration section in fixed metric order
    print("\n=== Calibration Metrics († = significantly worse than MENSA, Holm α=0.05) ===\n")
    for dataset in dataset_names:
        for metric in calib_metrics_display:
            sub = res_df[(res_df["Dataset"] == dataset) & (res_df["Metric"] == metric)]
            if sub.empty:
                continue
            worse_models = sub.loc[sub["worse_than_mensa"], "Model"].tolist()
            tag = " †" if worse_models else ""
            print(f"{dataset:<12} | {metric:<7} | Worse: {', '.join(worse_models)}{tag}")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    df = load_results()
    res = analyze(df)
    print_results(res)
