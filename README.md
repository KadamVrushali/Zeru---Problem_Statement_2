#  🌟 Wallet Risk Scoring From Scratch


# 💼 Wallet Risk Scoring – Compound V2/V3

This project calculates **risk scores (0–1000)** for Ethereum wallets based on their transaction activity with the **Compound V2 or V3 protocol**.

---

## 📌 Problem Statement

Given tasks:

1. **Fetch** their DeFi transaction history from Compound V2/V3.
2. **Preprocess** the data and extract meaningful behavioral features.
3. **Assign a risk score (0–1000)** to each wallet based on activity, volume, volatility, and interaction type.

---

## 🛠️ Steps Followed

### 1. **Data Collection**

* Used **Etherscan API** to collect Ethereum transaction data.
* Filtered transactions related to **Compound V2/V3** contracts only.
* Saved raw and cleaned data to JSON files.

### 2. **Data Preprocessing**

* Converted raw transactions to a uniform format: `deposit`, `borrow`, `repay`, `redeemUnderlying`, `liquidationCall`.
* Extracted wallet-level features such as:

  * Number of transactions
  * Failed transaction count
  * Total ETH moved
  * Mean and variance in transaction amounts
  * Number of unique counterparties
  * Function types used and frequency
  * Average time between transactions
  * High-volume bursts in short time (bot-like behavior)

### 3. **Risk Scoring**

* Used a custom scoring logic based on normalized features.
* Risk increases with:

  * Failed transactions
  * Large sudden volume
  * Liquidation-related actions
* Risk decreases with:

  * Stable, predictable behavior
  * Low volatility
* Final score normalized to a **0–1000** scale.

---

## 📁 Output Files

* `formatted_transactions.json`: Cleaned transaction list in unified schema.
* `wallet_features.csv`: Feature data for each wallet.
* `wallet_scores.csv`: Final wallet scores with columns: `wallet_id, score`.

---

## 🚀 How to Run


1. **All Steps:**

   ```bash
   python assignment_2.py
   ```

2. **Output csv file:**

   ```bash
   python wallet_risk_csv.py
   ```

---

## 📊 Example Output

```
wallet_id,score
0xfaa0768bde629806739c3a4620656c5d26f44ef2,732
0xabc12345abcd6789efab1234cd5678abcd987654,201
...
```


## ✍️ Author

Vrushali Kadam – Research Intern | AI + Blockchain

GitHub: https://github.com/kadamvrushali

LinkedIn: https://www.linkedin.com/in/vrushalikadam14

---

