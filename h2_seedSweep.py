import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.model import QKDGatekeeperLSTM
# ------------------------------
# experimental setup parameters
# ------------------------------
SEEDS = [0, 1, 2, 3, 4] 
WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.8 # 80% training, 20% test
CSV_PATH = "data/qkd_simulated_telemetry.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_windows(df, window_size, scaler=None, is_train=True):
    # extracting the physical features and the ground truth metadata
    X = df[["qber", "jitter", "dark_counts"]].to_numpy()
    y = df["attack_label"].to_numpy()
    stealthy = df["is_stealthy"].to_numpy()
    intensity = df["attack_intensity"].to_numpy()

    # scaler fitted only on training data, then applied to test data
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_seq, y_seq, stealthy_seq, intensity_seq = [], [], [], []
    # sliding the window across the dataset to create sequences for LSTM input
    for i in range(0, len(X_scaled) - window_size):
        X_seq.append(X_scaled[i: i + window_size])
        y_seq.append(y[i + window_size - 1])
        stealthy_seq.append(stealthy[i + window_size - 1])
        intensity_seq.append(intensity[i + window_size - 1])

    return (np.array(X_seq), np.array(y_seq), np.array(stealthy_seq),
            np.array(intensity_seq), scaler)


def recall_on_subset(preds, y_true, mask):
    # isolate predictions and ground truths for the specified subset
    subset_true = y_true[mask]
    subset_pred = preds[mask]
    # prevent division by zero if there are no true positives in the subset
    if len(subset_true) == 0:
        return np.nan, 0
    return (subset_pred == 1).mean(), len(subset_true) # return recall and number of samples in the subset


def train_lstm(X_train, y_train, seed):
    # seeds for robustness and reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # arrays turned into tensors 
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    
    g = torch.Generator()# same as above for the generator
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    #initialize the LSTM model, loss function, and optimiser
    model = QKDGatekeeperLSTM(input_size=3, hidden_dim=64, num_layers=2).to(DEVICE)
    #BCE
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    # training loop for the LSTM model
    for _ in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimiser.step()

    return model


if __name__ == "__main__":
    # load dataset and split into training and test sets
    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    # build windows for training and testing, scaling the features based on the training set
    X_train, y_train, _, _, scaler = build_windows(train_df, WINDOW_SIZE, is_train=True)
    X_test, y_test, stealthy_test, intensity_test, _ = build_windows(
        test_df, WINDOW_SIZE, scaler=scaler, is_train=False
    )

    print(f"Test windows: {len(y_test)} | attacks: {int(y_test.sum())} "
          f"(stealthy: {int(((y_test==1)&(stealthy_test)).sum())}, "
          f"non-stealthy: {int(((y_test==1)&(~stealthy_test)).sum())})")

    stealthy_mask = (y_test == 1) & stealthy_test
    nonstealthy_mask = (y_test == 1) & (~stealthy_test)

    # flatten the training and test data for the baseline models
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    rows = []
    # loop through the seeds for baselines and LSTM training on the same dataset
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")

        # ---- LSTM ----
        model = train_lstm(X_train, y_train, seed)
        model.eval()
        with torch.no_grad():
            lstm_probs = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        lstm_preds = (lstm_probs >= 0.5).astype(int)

        # ---- Baselines ----
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf.fit(X_train_flat, y_train)
        rf_preds = rf.predict(X_test_flat)

        lr = LogisticRegression(max_iter=1000, random_state=seed)
        lr.fit(X_train_flat, y_train)
        lr_preds = lr.predict(X_test_flat)
        
        # compare the recall on stealthy and non-stealthy attacks for each model
        for name, preds in [("LSTM", lstm_preds), ("RF", rf_preds), ("LR", lr_preds)]:
            r_stealthy, n_s = recall_on_subset(preds, y_test, stealthy_mask)
            r_nonstealthy, n_ns = recall_on_subset(preds, y_test, nonstealthy_mask)
            print(f"  {name:4s} -> non-stealthy recall={r_nonstealthy:.4f} "
                  f"stealthy recall={r_stealthy:.4f} drop={r_nonstealthy - r_stealthy:+.4f}")
            # append the results for this seed and model to the rows list for later aggregation
            rows.append({
                "seed": seed, "model": name,
                "recall_nonstealthy": r_nonstealthy, "recall_stealthy": r_stealthy,
                "recall_drop": r_nonstealthy - r_stealthy,
                "n_nonstealthy": n_ns, "n_stealthy": n_s,
            })

    # turn results into a DataFrame and save to CSV 
    results_df = pd.DataFrame(rows)
    results_df.to_csv("h2_seed_sweep_results.csv", index=False)

    print("\n" + "==================" )
    print("H2 SUMMARY (mean +/- SD across 5 seeds, same fixed dataset)")
    print("==================")
    # calculate mean and standard deviation of recall metrics across seeds for each model
    summary = results_df.groupby("model")[["recall_nonstealthy", "recall_stealthy", "recall_drop"]].agg(["mean", "std"])
    print(summary)
    summary.to_csv("h2_seed_sweep_summary.csv")

    # paired t-tests to compare LSTM vs baselines on stealthy recall across seeds and save results to CSV
    print("\nPaired t-tests (LSTM vs baseline by seed, on stealthy recall):")
    sig_rows = []
    lstm_recall = results_df[results_df.model == "LSTM"].sort_values("seed")["recall_stealthy"].values
    for name in ["RF", "LR"]:
        other_recall = results_df[results_df.model == name].sort_values("seed")["recall_stealthy"].values
        t_stat, p_val = stats.ttest_rel(lstm_recall, other_recall)
        print(f"  LSTM vs {name}: mean diff = {lstm_recall.mean() - other_recall.mean():+.4f}, "
              f"t={t_stat:.3f}, p={p_val:.6f}")
        sig_rows.append({"comparison": f"LSTM vs {name}", "metric": "recall_stealthy",
                          "mean_diff": lstm_recall.mean() - other_recall.mean(),
                          "t_stat": t_stat, "p_value": p_val})

    pd.DataFrame(sig_rows).to_csv("h2_seed_sweep_significance.csv", index=False)

    print("\nSaved h2_seed_sweep_results.csv (raw), h2_seed_sweep_summary.csv (aggregated),")
    print("and h2_seed_sweep_significance.csv (paired t-tests)")