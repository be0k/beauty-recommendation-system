import pandas as pd
import math

def get_date_info(all_review):
    # Converte reviewtime into datetime
    all_review["reviewTime"] = pd.to_datetime(all_review["reviewTime"], errors="coerce")

    # Print Oldest Date, and Latest Date
    min_date = all_review["reviewTime"].min()
    max_date = all_review["reviewTime"].max()

    print("Most oldest review date:", min_date)
    print("Most latest review date:", max_date)


def recall_at_k(y_true, y_pred):
    y_true, y_pred = set(y_true), set(y_pred)
    if len(y_true) == 0:
        return 0.0
    hit = len(y_true & y_pred)
    return hit / len(y_true)


def ndcg_at_k(y_true, y_pred):
    y_true = set(y_true)
    dcg = 0.0
    for i, item in enumerate(y_pred):
        if item in y_true:
            dcg += 1 / math.log2(i + 2)

    ideal_hits = min(len(y_true), len(y_pred))
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0