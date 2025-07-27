import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

def get_address_full_transaction_history(address: str):
    api_key = "API_KEY"  # replace with your Etherscan API key
    full_transactions = []
    next_block = 0

    while True:
        url = (
            f"https://api.etherscan.io/v2/api"
            f"?chainid=1"
            f"&module=account"
            f"&action=txlist"
            f"&address={address}"
            f"&startblock={next_block}"
            f"&endblock=latest"
            f"&page=1"
            f"&offset=1000"
            f"&sort=asc"
            f"&apikey={api_key}"
        )

        response = requests.get(url)
        data = response.json()

        if "result" not in data or len(data["result"]) == 0:
            break  # no more transactions

        txs = data["result"]

        # update next block
        last_block = int(txs[-1]["blockNumber"])
        next_block = last_block

        # add transactions
        for tx in txs:
            # only add until last block or if batch less than 1000
            if int(tx["blockNumber"]) != last_block or len(txs) != 1000:
                full_transactions.append(tx)

        # break if this was the last page
        if len(txs) < 1000:
            break
        
    print(f"Retrieved {len(full_transactions)} transactions for {address}")
    return full_transactions

def feature_extract_wallet(transactions: dict, wallet_id: str):
    df = transactions[wallet_id]

    # Handle no transactions
    if df.empty:
        return {
            "wallet_id": wallet_id,
            "num_transactions": 0,
            "num_failed": 0,
            "total_value_eth": 0,
            "mean_value_eth": 0,
            "std_value_eth": 0,
            "unique_function_calls": 0,
        }

    # --- Filter successful tx with non-empty functionName ---
    df = df[df["isError"] == "0"]
    df = df[df["functionName"].str.strip() != ""]
    if df.empty:
        return {
            "wallet_id": wallet_id,
            "num_transactions": 0,
            "num_failed": (transactions[wallet_id]["isError"] != "0").sum(),
            "total_value_eth": 0,
            "mean_value_eth": 0,
            "std_value_eth": 0,
            "unique_function_calls": 0,
        }

    # Convert numeric fields
    df["value"] = df["value"].astype(float)
    df["timeStamp"] = pd.to_datetime(df["timeStamp"].astype(int), unit="s")

    # --- Basic stats ---
    num_txs = len(df)
    num_failed = (transactions[wallet_id]["isError"] != "0").sum()

    # Values in ETH
    values_eth = df["value"] / 1e18
    total_value_eth = values_eth.sum()
    mean_value_eth = values_eth.mean()
    std_value_eth = values_eth.std(ddof=0)  # population std deviation

    # Unique counterparties
    unique_counterparties = len(set(df["from"]).union(set(df["to"])))

    # Function call stats
    func_counts = df["functionName"].value_counts()
    unique_functions = len(func_counts)
    top_function = func_counts.idxmax()

    # Time differences
    time_diffs = df["timeStamp"].sort_values().diff().dt.total_seconds().dropna()
    avg_time_gap = time_diffs.mean() if not time_diffs.empty else np.nan

    return {
        "wallet_id": wallet_id,
        "num_transactions": num_txs,
        "num_failed": int(num_failed),
        "total_value_eth": total_value_eth,
        "mean_value_eth": mean_value_eth,
        "std_value_eth": std_value_eth,
        "unique_counterparties": unique_counterparties,
        "unique_function_calls": unique_functions,
        "most_common_function": top_function,
        "avg_time_between_txs_sec": avg_time_gap,
    }


wallet_ids = pd.read_csv("walletid.csv")["wallet_id"].tolist()
transactions = {k: pd.DataFrame(get_address_full_transaction_history(k)) for k in wallet_ids}
features_list = [feature_extract_wallet(transactions, w) for w in wallet_ids]
features_df = pd.DataFrame(features_list)

# numeric features
numeric_cols = [c for c in features_df.columns if c not in ['wallet_id', 'most_common_function']]
X = features_df[numeric_cols].fillna(0)
X['total_value_eth'] = np.log1p(X['total_value_eth'])

# === MinMax Scaling ===
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)
# weights
weights = {
    "num_failed": 0.3,
    "total_value_eth": 0.25,
    "unique_counterparties": 0.2,
    "std_value_eth": 0.1,
    "num_transactions": 0.1,
    "avg_time_between_txs_sec": 0.05,
}
risk_raw = np.zeros(len(X_scaled))
for col, w in weights.items():
    risk_raw += X_scaled[col] * w

features_df["MM_risk_score"] = (risk_raw * 1000).astype(int)

# === Local Outlier Factor ===
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_scores = -lof.fit_predict(X_scaled)  # -1 for outlier
lof_factor = -lof.negative_outlier_factor_  # higher = more outlier

lof_risk = 1000 * (lof_factor - lof_factor.min()) / (lof_factor.max() - lof_factor.min())
features_df["lof_risk_score"] = lof_risk.astype(int)

# === Isolation Forest ===
# Fit isolation forest
iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_scaled)

# anomaly_score: the lower, the more anomalous
scores = -iso.score_samples(X_scaled)  # negative sign to make higher=more anomalous
# Scale to 0–1000
risk_score = 1000 * (scores - scores.min()) / (scores.max() - scores.min())
risk_score = risk_score.astype(int)

features_df["IF_risk_score"] = risk_score

# === Mixing Signals ===
features_df['risk_score'] = features_df[['MM_risk_score', 'lof_risk_score', 'IF_risk_score']].mean(axis=1).astype(int)

print(features_df[['wallet_id', 'risk_score', 'MM_risk_score', 'lof_risk_score', 'IF_risk_score']].head())

features_df[['wallet_id', 'risk_score']].to_csv("wallet_risk_scores.csv", index=False)
