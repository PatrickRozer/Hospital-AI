# src/association/run_association.py
import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from src.association.preprocess_association import load_and_prepare_association

MODEL_DIR = os.path.join("models", "association")
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # 1. Load data
    df = load_and_prepare_association()
    transactions = df["items"].tolist()

    if not transactions or all(len(t)==0 for t in transactions):
        print("⚠️ No transactions found. Check DIAGNOSES_ICD / PROCEDURES_ICD / PRESCRIPTIONS CSVs.")
        return

    # 2. One-hot encode
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    # 3. Run Apriori
    frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)  # lowered min_support
    if frequent_itemsets.empty:
        print("⚠️ No frequent itemsets found. Try lowering min_support further.")
    else:
        frequent_itemsets.to_csv(os.path.join(MODEL_DIR, "frequent_itemsets.csv"), index=False)
        print(f"✅ Saved frequent_itemsets.csv with {len(frequent_itemsets)} rows")

    # 4. Generate rules
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        if rules.empty:
            print("⚠️ No association rules found with lift >= 1.0")
        else:
            rules_sorted = rules.sort_values(by="lift", ascending=False)
            rules_sorted.to_csv(os.path.join(MODEL_DIR, "association_rules.csv"), index=False)
            print(f"✅ Saved association_rules.csv with {len(rules_sorted)} rows")
            print(rules_sorted.head(10))

if __name__ == "__main__":
    main()
