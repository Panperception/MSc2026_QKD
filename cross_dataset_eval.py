import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data_gen import generate_telemetry_dataset
from src.model import QKDGatekeeperLSTM

# -----------------------------------
# cross-dataset experimental setup
# -----------------------------------

N_TRAIN = 10                     #number of datasets pooled for training
N_TEST = 8                         #  number of datasets tested on separately
TRAIN_SEEDS = list(range(0, N_TRAIN))
TEST_SEEDS = list(range(N_TRAIN, N_TRAIN + N_TEST))
WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
MODEL_SEED = 42                    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def window_dataset(df, window_size, scaler):
    # extract features and labels from the dataframe and create sliding windows for LSTM input
    X = df[["qber", "jitter", "dark_counts"]].to_numpy()
    y = df["attack_label"].to_numpy()
    stealthy = df["is_stealthy"].to_numpy()

    # apply the scaler fitted on the training data to the test data
    X_scaled = scaler.transform(X)

    # create sliding windows for LSTM input and corresponding labels
    X_seq, y_seq, stealthy_seq = [], [], []
    for i in range(0, len(X_scaled) - window_size):
        X_seq.append(X_scaled[i: i + window_size])
        y_seq.append(y[i + window_size - 1])
        stealthy_seq.append(stealthy[i + window_size - 1])

    return np.array(X_seq), np.array(y_seq), np.array(stealthy_seq)


def recall_on_subset(preds, y_true, mask):
    # isolate predictions and ground truths for the specified subset
    pred_labels = preds[mask]
    true_labels = y_true[mask]
    # prevent division by zero if there are no true positives in the subset
    if mask.sum() == 0:
        return np.nan, 0
    return (pred_labels == 1).mean(), len(true_labels)


def paired_tests(df_wide, metric_name, baseline_names=("RF", "LR")):
   # perform paired t-tests between the LSTM model and baseline models
    results = []
    lstm_vals_full = df_wide["LSTM"].values
    for name in baseline_names:
        other_vals_full = df_wide[name].values
        # filter out any NaN values from both LSTM and baseline model results
        valid = ~(np.isnan(lstm_vals_full) | np.isnan(other_vals_full))
        n_used = int(valid.sum())
        lstm_vals = lstm_vals_full[valid]
        other_vals = other_vals_full[valid]

        if n_used < 2:
            results.append({"metric": metric_name, "comparison": f"LSTM vs {name}",
                             "n_test_sets": n_used, "mean_diff": np.nan,
                             "t_stat": np.nan, "t_p": np.nan})
            continue
        
        # perform paired t-test 
        t_stat, t_p = stats.ttest_rel(lstm_vals, other_vals)

        results.append({"metric": metric_name, "comparison": f"LSTM vs {name}",
                         "n_test_sets": n_used,
                         "mean_diff": lstm_vals.mean() - other_vals.mean(),
                         "t_stat": t_stat, "t_p": t_p})
    return results


if __name__ == "__main__":
    print(f"Training datasets: seeds {TRAIN_SEEDS}")
    print(f"Test datasets: seeds {TEST_SEEDS}\n")

    # generate and pool training datasets, fit scaler, and train LSTM and baseline models    
    train_dfs = [generate_telemetry_dataset(seed=s) for s in TRAIN_SEEDS]
    pooled_train_X = np.concatenate(
        [df[["qber", "jitter", "dark_counts"]].to_numpy() for df in train_dfs], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(pooled_train_X)

    X_train_parts, y_train_parts = [], []
    # create sliding windows for each training dataset and pool them together
    for df in train_dfs:
        X_seq, y_seq, _ = window_dataset(df, WINDOW_SIZE, scaler)
        X_train_parts.append(X_seq)
        y_train_parts.append(y_seq)
    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    print(f"Pooled training windows: {len(X_train)} (from {N_TRAIN} datasets)\n")

    
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)

    # convert training data to PyTorch tensors and create a DataLoader for LSTM training
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    g = torch.Generator()
    g.manual_seed(MODEL_SEED)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    model = QKDGatekeeperLSTM(input_size=3, hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train the LSTM model on the pooled training datasets
    print("Training LSTM on training set")
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss:.5f}")
    model.eval()

    #flatten the training data for the baseline models
    X_train_flat = X_train.reshape(len(X_train), -1)
    print("\nTraining Random Forest and Logistic Regression on the same pooled data...")
    rf = RandomForestClassifier(n_estimators=100, random_state=MODEL_SEED, n_jobs=-1)
    rf.fit(X_train_flat, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=MODEL_SEED)
    lr.fit(X_train_flat, y_train)

    
    agg_rows = []
    strat_rows = []

    # evaluate the trained models on each test dataset and collect metrics
    for test_seed in TEST_SEEDS:
        print(f"\nEvaluating on test dataset (seed {test_seed})")
        test_df = generate_telemetry_dataset(seed=test_seed)
        X_test, y_test, stealthy_test = window_dataset(test_df, WINDOW_SIZE, scaler)
        X_test_flat = X_test.reshape(len(X_test), -1)

        # generate LSTM predictions and probabilities
        with torch.no_grad():
            lstm_probs = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        lstm_preds = (lstm_probs >= 0.5).astype(int)

        rf_preds = rf.predict(X_test_flat)
        rf_probs = rf.predict_proba(X_test_flat)[:, 1]

        lr_preds = lr.predict(X_test_flat)
        lr_probs = lr.predict_proba(X_test_flat)[:, 1]

        # calculate the stealthy and non-stealthy masks for stratified evaluation
        stealthy_mask = (y_test == 1) & stealthy_test
        nonstealthy_mask = (y_test == 1) & (~stealthy_test)

        for name, probs, preds in [("LSTM", lstm_probs, lstm_preds),
                                    ("RF", rf_probs, rf_preds),
                                    ("LR", lr_probs, lr_preds)]:
            # calculate and print aggregated metrics for accuracy, F1-score, and ROC-AUC
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            print(f"  {name:4s} -> acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
            agg_rows.append({"test_seed": test_seed, "model": name,
                              "accuracy": acc, "f1": f1, "auc": auc})

            # calculate and print recall metrics for stealthy and non-stealthy attacks
            r_stealthy, n_s = recall_on_subset(preds, y_test, stealthy_mask)
            r_nonstealthy, n_ns = recall_on_subset(preds, y_test, nonstealthy_mask)
            print(f"  {name:4s} -> non-stealthy recall={r_nonstealthy:.4f} (n={n_ns}) "
                  f"stealthy recall={r_stealthy:.4f} (n={n_s})")
            strat_rows.append({"test_seed": test_seed, "model": name,
                                "recall_nonstealthy": r_nonstealthy, "recall_stealthy": r_stealthy,
                                "recall_drop": r_nonstealthy - r_stealthy,
                                "n_nonstealthy": n_ns, "n_stealthy": n_s})

    # save the aggregated and stratified results to CSV files and print summary statistics
    agg_df = pd.DataFrame(agg_rows)
    strat_df = pd.DataFrame(strat_rows)
    agg_df.to_csv("cross_dataset_sweep_aggregate.csv", index=False)
    strat_df.to_csv("cross_dataset_sweep_stratified.csv", index=False)

    print("\n" + "==================")
    print(f"H1 SUMMARY (mean +/- SD across {N_TEST}  test datasets)")
    print("==================")
    # calculate mean and standard deviation of accuracy, F1-score, and ROC-AUC across test datasets for each model
    agg_summary = agg_df.groupby("model")[["accuracy", "f1", "auc"]].agg(["mean", "std"])
    print(agg_summary)
    agg_summary.to_csv("cross_dataset_sweep_aggregate_summary.csv")

    print("\n"+ "==================")
    print(f"H2 SUMMARY (mean +/- SD across {N_TEST}  test datasets)")
    print("==================")
    strat_summary = strat_df.groupby("model")[["recall_nonstealthy", "recall_stealthy", "recall_drop"]].agg(["mean", "std"])
    print(strat_summary)
    strat_summary.to_csv("cross_dataset_sweep_stratified_summary.csv")


    auc_wide = agg_df.pivot(index="test_seed", columns="model", values="auc")
    h1_results = paired_tests(auc_wide, "auc")

    
    stealthy_wide = strat_df.pivot(index="test_seed", columns="model", values="recall_stealthy")
    h2_results = paired_tests(stealthy_wide, "recall_stealthy")

    pd.DataFrame(h1_results + h2_results).to_csv("cross_dataset_sweep_significance.csv", index=False)

    print("\n" + "==================")
    print(f"H1 - Paired t-test on ROC-AUC (LSTM vs baselines)")
    print("==================")
    for r in h1_results:
        print(f"  {r['comparison']} (n={r['n_test_sets']}): mean diff = {r['mean_diff']:+.4f} | "
              f"t-test p={r['t_p']:.4f}")

    print("\n" + "==================")
    print(f"H2 - Paired t-test on stealthy recall ")
    print("==================")
    for r in h2_results:
        print(f"  [{r['metric']}] {r['comparison']} (n={r['n_test_sets']}): mean diff = {r['mean_diff']:+.4f} | "
              f"t-test p={r['t_p']:.4f}")

    print("\nSaved cross_dataset_sweep_aggregate.csv, cross_dataset_sweep_stratified.csv,")
    print("       cross_dataset_sweep_aggregate_summary.csv, cross_dataset_sweep_stratified_summary.csv,")
    print("       cross_dataset_sweep_significance.csv")