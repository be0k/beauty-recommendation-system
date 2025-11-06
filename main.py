from src.utils import (getDF, 
                       get_user_item_matrix,
                       predict_user_ratings,
                       get_similarity_matrix_tfidf,
                       get_similarity_matrix_bm25,
                       get_similarity_matrix_bert,
                       recommend_trend_based,
                       get_similarity_recommendations,
                       get_similar_users_from_reviews,
                       get_user_embeds

                    )
from src.extra import (get_date_info,
                       recall_at_k,
                       ndcg_at_k
                       )
from src.preprocessing import preprocessing
import pandas as pd
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import warnings
warnings.filterwarnings('ignore')



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--test', type=str, required=True, help='Enter the test mode')
    parser.add_argument('--mode', type=str, required=False, help='Enter the mode')
    parser.add_argument('--sim', type=str, default='bert', required=True, help='one of "bm25", "tfidf", "bert"')
    parser.add_argument('--k', type=int, default=10, required=False, help='The number of k')
    parser.add_argument('--month', type=int, default=3, required=False, help='The number of month')
    parser.add_argument('--m', type=int, default=10, required=False, help='The number of m')
    # parser.add_argument('--file_name', type=str, required=True, help='Enter the file location')
    # parser.add_argument('--n_fold', type=int, default=10, required=False, help='The number of folds')
    # parser.add_argument('--tune', action='store_true', required=False, help='whether you do hyperparameter tuning')
    # parser.add_argument('--stratify', action='store_true', required=False, help='whether you use StratifiedKFold instead of KFold')
    # parser.add_argument('--iteration', type=int, default=30, required=False, help='hyperparameter tuning iteration')
    # parser.add_argument('--eda', action='store_true', required=False, help='whether you do EDA')
    # parser.add_argument('--onehot', action='store_true', required=False, help='whether you use one hot encoder')
    # parser.add_argument('--scaler', type=str, default=None, required=False, help='decide what scaler you use')
    # parser.add_argument('--seed', type=int, default=202334441, required=False, help='seed number')
    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    ################Data Load############
    all_review = getDF(DATA_DIR / "All_Beauty_5.json.gz")
    all_meta = getDF(DATA_DIR / 'meta_All_Beauty.json.gz')
    with open(DATA_DIR / "all_fill.json", "r", encoding="utf-8") as f:
        all_fill = json.load(f)

    lxr_review = getDF(DATA_DIR / "Luxury_Beauty_5.json.gz")
    lxr_meta = getDF(DATA_DIR / 'meta_Luxury_Beauty.json.gz')
    with open(DATA_DIR / "luxury_fill.json", "r", encoding="utf-8") as f:
        lxr_fill = json.load(f)
    

    ######################Preprocessing#######################
    all_review, all_meta = preprocessing(all_review, all_meta, all_fill, DATA_DIR)
    lxr_review, lxr_meta = preprocessing(lxr_review, lxr_meta, lxr_fill, DATA_DIR)
    rcmd_list = all_meta['asin'].unique().tolist()
    base_list = lxr_meta['asin'].unique().tolist()
    all_review["reviewTime"] = pd.to_datetime(all_review["reviewTime"], errors="coerce")
    lxr_review["reviewTime"] = pd.to_datetime(lxr_review["reviewTime"], errors="coerce")
    # print(all_meta.info())
    # print(lxr_meta.info())
    # print(all_review.info())
    # print(lxr_review.info())

    #get_date_info(all_review)

    ########################Similarity############################
    if args.sim=='bert':
        meta_sim = get_similarity_matrix_bert(all_meta, lxr_meta)
    elif args.sim=='tfidf':
        meta_sim = get_similarity_matrix_tfidf(all_meta, lxr_meta)
    elif args.sim=='bm25':
        meta_sim = get_similarity_matrix_bm25(all_meta, lxr_meta)
    else:
        print("similarity name is wrong")
        quit()
    # meta_sim.to_csv('sim.csv')

    #########################User-Item Matrix######################
    lxr_user_item = get_user_item_matrix(lxr_review, meta_sim)
    # lxr_user_item.to_csv('user_item.csv')

    ###########################Fill Cross Domain Part##############3..........??
    # predicted_user_item = predict_user_ratings(lxr_user_item, meta_sim)
    # predicted_user_item.to_csv('pseudo.csv', index=False)

    k = args.k
    m = args.m

    if args.test=='zero':
        recalls = []
        ndcgs = []
        for y, date in tqdm(zip(all_review['asin'], all_review['reviewTime']), total=len(all_review)):
            # recent_df = recommend_trend_based(meta_sim, lxr_review, rcmd_list, date=lxr_review['reviewTime'].max(), month=None, k=k)
            # Overall Trend
            if args.month==0:
                args.month = None
            recent_df = recommend_trend_based(meta_sim, lxr_review, rcmd_list, date=date, month=args.month, k=k)
            recalls.append(recall_at_k([y], list(recent_df['asin'])))
            ndcgs.append(ndcg_at_k([y], list(recent_df['asin'])))

        print(f'Recall@{k}: {np.mean(recalls):.4f}, nDCG@{k}: {np.mean(ndcgs):.4f}')

    elif args.test=='cold':

        recalls = []
        ndcgs = []

        all_review = all_review.sort_values(["reviewerID", "reviewTime"])

        first_reviews = all_review.groupby("reviewerID").first().reset_index()
        later_reviews = all_review.merge(first_reviews[["reviewerID", "asin"]], 
                                on="reviewerID", 
                                suffixes=("", "_first"))
        later_reviews = later_reviews[later_reviews["asin"] != later_reviews["asin_first"]]


        if args.mode=='col':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("intfloat/e5-large-v2", device=device)
            user_embeds = get_user_embeds(lxr_review, model)


        for bought, text, id in tqdm(zip(first_reviews['asin'], first_reviews['reviewText'], first_reviews['reviewerID']), total=len(first_reviews)):

            if args.mode=='content':
                recent_df = get_similarity_recommendations(meta_sim, [bought], rcmd_list, k=k, weight=None)
            else:
                target_emb = model.encode([text], normalize_embeddings=True)

                if bought in rcmd_list:
                    rcmd_list.remove(bought)

                recent_df = get_similar_users_from_reviews(lxr_user_item,
                                                            target_emb,
                                                            user_embeds,
                                                            base_list,
                                                            rcmd_list,
                                                            meta_sim,
                                                            k=k,
                                                            m=m
                                   )
                


            y = later_reviews[later_reviews["reviewerID"] == id]['asin'].tolist()
            recalls.append(recall_at_k(y, list(recent_df['asin'])))
            ndcgs.append(ndcg_at_k(y, list(recent_df['asin'])))

        print(f'Recall@{k}: {np.mean(recalls):.4f}, nDCG@{k}: {np.mean(ndcgs):.4f}')



    elif args.test=='warm':

        recalls = []
        ndcgs = []

        # reviewerID별 reviewTime 순으로 정렬
        all_review = all_review.sort_values(["reviewerID", "reviewTime"])

        # 각 유저의 최신 리뷰(latest)를 가져옴
        latest_reviews = all_review.groupby("reviewerID").last().reset_index()

        # 최신 리뷰를 Y로, 그 이전 리뷰들을 X로 구성
        earlier_reviews = all_review.merge(latest_reviews[["reviewerID", "asin"]], 
                                        on="reviewerID", 
                                        suffixes=("", "_latest"))

        # 최신 리뷰(asn_latest)만 제외하고 나머지를 X로 사용
        earlier_reviews = earlier_reviews[earlier_reviews["asin"] != earlier_reviews["asin_latest"]]

        if args.mode=='col':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer("intfloat/e5-large-v2", device=device)
            user_embeds = get_user_embeds(lxr_review, model)


        for y, id in tqdm(zip(latest_reviews['asin'], latest_reviews['reviewerID']), total=len(latest_reviews)):
            boughts = earlier_reviews[earlier_reviews["reviewerID"] == id]['asin'].tolist()

            if args.mode=='content':
                recent_df = get_similarity_recommendations(meta_sim, boughts, rcmd_list, k=k, weight=None)

            else:
                text = earlier_reviews[earlier_reviews["reviewerID"] == id]['reviewText'].tolist()
                text = ' '.join(text)
                target_emb = model.encode([text], normalize_embeddings=True)
                
                rcmd_list = [item for item in rcmd_list if item not in boughts]


                recent_df = get_similar_users_from_reviews(lxr_user_item,
                                   target_emb,
                                   user_embeds,
                                   base_list,
                                   rcmd_list,
                                   meta_sim,
                                   k=k,
                                   m=m)
                


            recalls.append(recall_at_k([y], list(recent_df['asin'])))
            ndcgs.append(ndcg_at_k([y], list(recent_df['asin'])))

        print(f'Recall@{k}: {np.mean(recalls):.4f}, nDCG@{k}: {np.mean(ndcgs):.4f}')


    else:




        # 1 Trend based recommendation
        recent_df = recommend_trend_based(meta_sim, lxr_review, rcmd_list, date="2015-07-01", month=3, k=k)
        recent_df.to_csv('trend.csv',index=False)


        # 2 Search similar stuff that specific user bought
        # second parameter = source_asins = the list of items that user bought
        # third parameter = candidate_asins = the list of items that you set the range
        recent_df = get_similarity_recommendations(meta_sim, rcmd_list[:5], rcmd_list, k=k, weight=None)
        recent_df.to_csv('content.csv',index=False)

        # 3 Collaborative filtering
        good = get_similar_users_from_reviews(lxr_review,
                                    lxr_user_item,
                                    "It is sooooo goooooood.",
                                    base_list,#all re
                                    rcmd_list,
                                    meta_sim,
                                    k=10
                                    )
        good.to_csv('collaborative.csv')

if __name__=='__main__':
    main()