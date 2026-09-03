import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy import stats

from src.dataset import get_qkd_dataloaders
from src.model import QKDGatekeeperLSTM

# -----------------------------------
# experimental setup parameters
# -----------------------------------

SEEDS = [0, 1, 2, 3, 4]         
WINDOW_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_lstm(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load the dataloaders for the LSTM model with the specified seed
    train_loader, test_loader, scaler = get_qkd_dataloaders(
        window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, seed=seed
    )
    # initialise the LSTM model, loss function, and optimiser
    model = QKDGatekeeperLSTM(input_size=3, hidden_dim=64, num_layers=2).to(DEVICE)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop for the LSTM model
    model.train()
    for _ in range(EPOCHS):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimiser.step()

    # evaluation of the trained LSTM model on the test set
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            batch_probs = model(X_batch.to(DEVICE))
            probs.extend(batch_probs.cpu().numpy())
            targets.extend(y_batch.numpy())
    probs, targets = np.array(probs), np.array(targets)
    
    # convert probabilities to binary predictions using a threshold of 0.5
    preds = (probs >= 0.5).astype(int)

    #return aggregated metrics for accuracy, F1-score, and ROC-AUC
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1": f1_score(targets, preds),
        "auc": roc_auc_score(targets, probs),
    }


def run_baselines(seed):
    # load the dataloaders for the baseline models with the specified seed
    train_loader, test_loader, scaler = get_qkd_dataloaders(
        window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, seed=seed
    )
    
    # convert the training and test data into flattened numpy arrays for the baseline models
    X_train = train_loader.dataset.X_seq.numpy().reshape(len(train_loader.dataset), -1)
    y_train = train_loader.dataset.y_seq.numpy()
    X_test = test_loader.dataset.X_seq.numpy().reshape(len(test_loader.dataset), -1)
    y_test = test_loader.dataset.y_seq.numpy()

    results = {}
    
    # train and evaluate the Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results["rf"] = {
        "accuracy": accuracy_score(y_test, rf_preds),
        "f1": f1_score(y_test, rf_preds),
        "auc": roc_auc_score(y_test, rf_probs),
    }
    # train and evaluate the Logistic Regression classifier
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    results["lr"] = {
        "accuracy": accuracy_score(y_test, lr_preds),
        "f1": f1_score(y_test, lr_preds),
        "auc": roc_auc_score(y_test, lr_probs),
    }
    return results


if __name__ == "__main__":
    
    rows = []
    #run, print and save the results for each seed and model to CSV
    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        lstm_res = run_lstm(seed)
        print(f"  LSTM  -> acc={lstm_res['accuracy']:.4f} f1={lstm_res['f1']:.4f} auc={lstm_res['auc']:.4f}")
        rows.append({"seed": seed, "model": "LSTM", **lstm_res})

        baseline_res = run_baselines(seed)
        for name, res in baseline_res.items():
            print(f"  {name.upper():4s} -> acc={res['accuracy']:.4f} f1={res['f1']:.4f} auc={res['auc']:.4f}")
            rows.append({"seed": seed, "model": name.upper(), **res})

    df = pd.DataFrame(rows)
    df.to_csv("seed_sweep_results.csv", index=False)

    print("\n" + "====================")
    print("SUMMARY (mean +/- SD across seeds)")
    print("====================")
    summary = df.groupby("model")[["accuracy", "f1", "auc"]].agg(["mean", "std"])
    print(summary)
    summary.to_csv("seed_sweep_summary.csv")

    # paired significance test for LSTM vs each baseline on same seeds
    print("\nPaired t-tests (LSTM vs baseline, matched by seed, on AUC):")
    lstm_auc = df[df.model == "LSTM"].sort_values("seed")["auc"].values
    for name in ["RF", "LR"]:
        other_auc = df[df.model == name].sort_values("seed")["auc"].values
        t_stat, p_val = stats.ttest_rel(lstm_auc, other_auc)
        print(f"  LSTM vs {name}: mean diff = {lstm_auc.mean()-other_auc.mean():+.4f}, "
              f"t={t_stat:.3f}, p={p_val:.4f}")

    print("\nSaved seed_sweep_results.csv and seed_sweep_summary.csv")