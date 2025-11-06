import pandas as pd
import gzip
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import torch

#For Loading Data (https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


#Get Co-item similarity based on TF-IDF
def get_similarity_matrix_tfidf(df1, df2):

    # 1 Merge Train & Test Items
    combined_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["asin"]).reset_index(drop=True)

    # 2 Concat Title and Description
    combined_df["text"] = (combined_df["title"].fillna("") + " " + combined_df["description"].fillna("")).str.strip()

    # 3 Train tf-idf
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined_df["text"])

    # 4 Get cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # 5 numpy --> DF
    similarity_df = pd.DataFrame(similarity_matrix, index=combined_df["asin"], columns=combined_df["asin"])

    return similarity_df


def get_similarity_matrix_bm25(df1, df2):
    # 1 Merge Train & Test Items
    combined_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=["asin"]).reset_index(drop=True)

    # 2 Concat Title and Description
    combined_df["text"] = (combined_df["title"].fillna("") + " " + combined_df["description"].fillna("")).str.strip()

    # 3 Tokenize
    corpus = [text.lower().split() for text in combined_df["text"].tolist()]

    # 4 Train BM25
    bm25 = BM25Okapi(corpus)

    n = len(corpus)
    similarity_matrix = np.zeros((n, n))

    for i in tqdm(range(n), desc="Computing BM25 similarities", leave=False):
        scores = bm25.get_scores(corpus[i])
        similarity_matrix[i] = scores

    # 5 Normalization
    similarity_matrix = similarity_matrix / similarity_matrix.max()

    # 6 Convert numpy into df
    similarity_df = pd.DataFrame(similarity_matrix, index=combined_df["asin"], columns=combined_df["asin"])

    return similarity_df

def get_similarity_matrix_bert(df1, df2, model_name = "all-MiniLM-L6-v2", batch_size: int = 64):

    # 1 Merge Train & Test Items
    combined_df = pd.concat([df1, df2], ignore_index=True)\
                    .drop_duplicates(subset=["asin"])\
                    .reset_index(drop=True)
    
    # 2 Concat Title and Description
    combined_df["text"] = (combined_df["title"].fillna("") + " " + combined_df["description"].fillna("")).str.strip()
    
    # 3 Load embedding model
    model = SentenceTransformer(model_name)
    texts = combined_df["text"].tolist()
    
    # 5 Generate Word Embedding
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    
    # 6 Get cos sim
    similarity_matrix = cosine_similarity(embeddings, embeddings)
    
    # 7 convert numpy into df
    similarity_df = pd.DataFrame(similarity_matrix,
                                 index=combined_df["asin"],
                                 columns=combined_df["asin"])
    return similarity_df


def get_user_item_matrix(review_df, similarity_df, only_buy=False):
    # 1. generate pivot-table (user-item matrix)
    user_item = review_df.pivot_table(
        index="reviewerID",
        columns="asin",
        values="overall",
        fill_value=0  # Fill 0 if user-item is nan
    )

    asins_in_similarity = similarity_df.columns.tolist()

    # Align column
    user_item = user_item.reindex(columns=asins_in_similarity, fill_value=0)
    
    if only_buy:
        user_item[:] = np.where(user_item.values != 0, 1, 0)

    # Check
    # print(user_item.shape)
    # print(user_item.head())

    return user_item

#안쓸듯
def predict_user_ratings(user_item: pd.DataFrame, meta_sim: pd.DataFrame):

    R = user_item.values.astype(float)
    S = meta_sim.values.astype(float)
    
    numerators = R @ S  # 각 유저 × 아이템의 (rating × similarity)
    denominators = np.abs((R > 0).astype(float) @ S)  # The sum of similarity absolute
    
    # Prevent that the number is divided by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        predictions = np.divide(numerators, denominators, where=denominators != 0)
        predictions[np.isnan(predictions)] = 0  # NaN is 0
    
    # Restruct to original ones.
    pred_df = pd.DataFrame(predictions, index=user_item.index, columns=user_item.columns)
    return pred_df



def get_current_trend(df, date="2015-07-01", month=3):
    # Text into datetime
    df["reviewTime"] = pd.to_datetime(df["reviewTime"], errors="coerce")

    # Standard date
    input_date = pd.Timestamp(date)

    # Calculate the date that is N months ago from standard date
    if month==None:
        start_date = df["reviewTime"].min()
    else:
        start_date = input_date - pd.DateOffset(months=month)

    # Filtering the data that the date is from start date to input date
    df = df[(df["reviewTime"] >= start_date) & (df["reviewTime"] <= input_date)]

    # Count The number of Review per item
    item_counts = df["asin"].value_counts().rename_axis("asin").reset_index(name="count")

    # Get Maximum count
    max_count = item_counts["count"].max()

    # Trend score calculation (https://arxiv.org/pdf/2509.13957)
    item_counts["strend"] = np.log(item_counts["count"] / max_count + 1)

    return item_counts


def recommend_trend_based(meta_sim: pd.DataFrame, 
                          review_df: pd.DataFrame, 
                          rcmd_list, 
                          date="2015-07-01", 
                          month=3, 
                          k=5):
    # Get Recent Trend
    trend_df = get_current_trend(review_df, date=date, month=month)

    # If Recent Trend doesnt exist, Use Overall Trend instead of Recent Trend.
    if len(trend_df) == 0:
        print("The recent trend don't exist. So, we consider the overall trend instead of this.")
        trend_df = get_current_trend(review_df, date=review_df["reviewTime"].max(), month=None)

    # source_asins set + trend score
    source_asins = trend_df["asin"].tolist()
    trend_scores = trend_df["strend"].tolist()

    # get_similarity_recommendations
    result = get_similarity_recommendations(
        meta_sim=meta_sim,
        source_asins=source_asins,
        candidate_asins=rcmd_list,
        k=k,
        weight=trend_scores
    )

    return result



def get_similarity_recommendations(meta_sim: pd.DataFrame, 
                                   source_asins, 
                                   candidate_asins, 
                                   k=5, 
                                   weight=None):

    # Filtering validate asins
    valid_sources = [a for a in source_asins if a in meta_sim.columns]
    valid_candidates = [a for a in candidate_asins if a in meta_sim.columns]
    if not valid_sources or not valid_candidates:
        return pd.DataFrame(columns=["asin", "score"])

    # Get row/column idx
    source_idx = [meta_sim.columns.get_loc(a) for a in valid_sources]
    candidate_idx = [meta_sim.columns.get_loc(a) for a in valid_candidates]

    # Extraction only nessesary part (shape: len(candidate) x len(source))
    sub_sim = meta_sim.iloc[candidate_idx, source_idx].to_numpy()

    # Reflection weight (If weight is None, Just use vanilla mean)
    if weight is not None:
        weight = np.array(weight).reshape(-1, 1)
        scores = (sub_sim @ weight).flatten()
    else:
        scores = sub_sim.mean(axis=1)

    # Except to items already bought by user 
    mask = ~pd.Series(valid_candidates).isin(valid_sources)
    scores = scores[mask]
    valid_candidates = np.array(valid_candidates)[mask]

    #Top-K Recommendation
    top_k_idx = np.argsort(scores)[::-1][:k]
    result = pd.DataFrame({
        "asin": [valid_candidates[i] for i in top_k_idx],
        "score": [scores[i] for i in top_k_idx]
    })

    return result


def get_user_embeds(review_df, model):
    # remove null and merge review texts per reviewer.
    df = review_df.dropna(subset=["reviewText"]).copy()
    user_reviews = df.groupby("reviewerID")["reviewText"].apply(lambda x: " ".join(x)).to_dict()

    user_ids = list(user_reviews.keys())
    texts = list(user_reviews.values())

    # User Embedding
    user_embeds = model.encode(texts, normalize_embeddings=True)
    return user_embeds

def top_M(vec, M):
    # sims를 복사해서 결과 배열 만들기
    filtered_sims = vec.copy()

    # top-N의 인덱스 구하기
    top_indices = np.argpartition(vec, -M)[-M:]  # 정렬 없이 top-N만 빠르게 찾음

    # 나머지 위치는 0으로 만듦
    mask = np.ones_like(vec, dtype=bool)
    mask[top_indices] = False
    filtered_sims[mask] = 0

    return filtered_sims

def get_similar_users_from_reviews(user_item: pd.DataFrame,
                                   target_emb,
                                   user_embeds,
                                   lxr_list,
                                   candidate_list,
                                   meta_sim,
                                   k: int = 5,
                                   m = 0
                                   ):

    # cosine similarity (n)
    sims = cosine_similarity(target_emb, user_embeds).flatten()

    #top-M selection
    if m!=0:
        sims = top_M(sims, m)

    # Only extraction items that is in lxr_list
    valid_lxr = [asin for asin in lxr_list if asin in user_item.columns]
    if not valid_lxr:
        raise ValueError("lxr_list 내에서 user_item에 존재하는 아이템이 없습니다.")

    sub_user_item = user_item[valid_lxr].to_numpy()  # (n_users, len(valid_lxr))

    # Weighted sum (only lxr list)
    weighted_items = sub_user_item.T @ sims
    sim_sums = (sub_user_item > 0).T @ sims
    item_sims = np.divide(weighted_items, sim_sums, out=np.zeros_like(weighted_items), where=sim_sums != 0)    

    result = get_similarity_recommendations(
        meta_sim=meta_sim,
        source_asins=valid_lxr,
        candidate_asins=candidate_list,
        k=k,
        weight=item_sims
    )

    return result
