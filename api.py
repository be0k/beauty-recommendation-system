from flask import Flask, request, jsonify
from src.utils import (
    getDF, 
    get_user_item_matrix,
    get_similarity_matrix_bert,
    recommend_trend_based,
    get_similarity_recommendations,
    get_similar_users_from_reviews,
    get_user_embeds
)
from src.preprocessing import preprocessing
import pandas as pd
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables for loaded data
DATA_CACHE = {}

def initialize_data():
    """Initialize and load all necessary data"""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    
    print("Loading data...")
    
    # Load data
    all_review = getDF(DATA_DIR / "All_Beauty_5.json.gz")
    all_meta = getDF(DATA_DIR / 'meta_All_Beauty.json.gz')
    with open(DATA_DIR / "all_fill.json", "r", encoding="utf-8") as f:
        all_fill = json.load(f)
    
    lxr_review = getDF(DATA_DIR / "Luxury_Beauty_5.json.gz")
    lxr_meta = getDF(DATA_DIR / 'meta_Luxury_Beauty.json.gz')
    with open(DATA_DIR / "luxury_fill.json", "r", encoding="utf-8") as f:
        lxr_fill = json.load(f)
    
    # Preprocessing
    print("Preprocessing...")
    all_review, all_meta = preprocessing(all_review, all_meta, all_fill, DATA_DIR)
    lxr_review, lxr_meta = preprocessing(lxr_review, lxr_meta, lxr_fill, DATA_DIR)
    
    all_review["reviewTime"] = pd.to_datetime(all_review["reviewTime"], errors="coerce")
    lxr_review["reviewTime"] = pd.to_datetime(lxr_review["reviewTime"], errors="coerce")
    
    rcmd_list = all_meta['asin'].unique().tolist()
    base_list = lxr_meta['asin'].unique().tolist()
    
    # Calculate similarity matrix
    print("Calculating similarity matrix...")
    meta_sim = get_similarity_matrix_bert(all_meta, lxr_meta)
    
    # Get user-item matrix
    print("Creating user-item matrix...")
    lxr_user_item = get_user_item_matrix(lxr_review, meta_sim)
    
    # Load BERT model for collaborative filtering
    print("Loading BERT model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("intfloat/e5-large-v2", device=device)
    user_embeds = get_user_embeds(lxr_review, model)
    
    # Create title mapping
    title_map = dict(zip(all_meta['asin'], all_meta['title']))
    
    # Cache everything
    DATA_CACHE['all_review'] = all_review
    DATA_CACHE['all_meta'] = all_meta
    DATA_CACHE['lxr_review'] = lxr_review
    DATA_CACHE['lxr_meta'] = lxr_meta
    DATA_CACHE['rcmd_list'] = rcmd_list
    DATA_CACHE['base_list'] = base_list
    DATA_CACHE['meta_sim'] = meta_sim
    DATA_CACHE['lxr_user_item'] = lxr_user_item
    DATA_CACHE['model'] = model
    DATA_CACHE['user_embeds'] = user_embeds
    DATA_CACHE['title_map'] = title_map
    
    print("Initialization complete!")


def rerank_recommendations(content_df, collab_df, trend_df, k=10, 
                          content_weight=0.4, collab_weight=0.4, trend_weight=0.2):
    """
    Rerank recommendations by combining multiple strategies
    
    Args:
        content_df: Content-based recommendations with scores
        collab_df: Collaborative filtering recommendations with scores
        trend_df: Trend-based recommendations with scores
        k: Number of final recommendations
        content_weight: Weight for content-based score
        collab_weight: Weight for collaborative score
        trend_weight: Weight for trend score
    
    Returns:
        DataFrame with reranked recommendations
    """
    # Normalize scores for each method (0-1 scale)
    def normalize_scores(df, score_col='score'):
        if len(df) == 0:
            return df
        df = df.copy()
        max_score = df[score_col].max()
        min_score = df[score_col].min()
        if max_score != min_score:
            df[score_col] = (df[score_col] - min_score) / (max_score - min_score)
        else:
            df[score_col] = 1.0
        return df
    
    content_df = normalize_scores(content_df)
    collab_df = normalize_scores(collab_df)
    trend_df = normalize_scores(trend_df)
    
    # Collect all unique items
    all_items = set()
    all_items.update(content_df['asin'].tolist())
    all_items.update(collab_df['asin'].tolist())
    all_items.update(trend_df['asin'].tolist())
    
    # Calculate combined scores
    combined_scores = {}
    for item in all_items:
        score = 0.0
        
        content_score = content_df[content_df['asin'] == item]['score']
        if len(content_score) > 0:
            score += content_weight * content_score.values[0]
        
        collab_score = collab_df[collab_df['asin'] == item]['score']
        if len(collab_score) > 0:
            score += collab_weight * collab_score.values[0]
        
        trend_score = trend_df[trend_df['asin'] == item]['score']
        if len(trend_score) > 0:
            score += trend_weight * trend_score.values[0]
        
        combined_scores[item] = score
    
    # Sort by combined score
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Create final DataFrame
    top_items = sorted_items[:k]
    result_df = pd.DataFrame({
        'asin': [item[0] for item in top_items],
        'score': [item[1] for item in top_items]
    })
    
    return result_df


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for recommendations
    
    Request JSON format:
    {
        "user_reviews": [
            {
                "asin": "item_id",
                "reviewText": "review text"
            },
            ...
        ],
        "k": 10,  # optional, default 10
        "m": 10,  # optional, default 10 (for collaborative filtering)
        "month": 3  # optional, default 3 (for trend-based)
    }
    
    Response JSON format:
    {
        "recommendations": [
            {
                "title": "Product Title",
                "asin": "item_id",
                "score": 0.85
            },
            ...
        ],
        "method": "trend" or "hybrid"
    }
    """
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400
        
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'error': 'Invalid JSON data'
            }), 400
        user_reviews = data.get('user_reviews', [])
        k = data.get('k', 10)
        m = data.get('m', 10)
        month = data.get('month', 3)
        
        # Get cached data
        meta_sim = DATA_CACHE['meta_sim']
        lxr_review = DATA_CACHE['lxr_review']
        lxr_user_item = DATA_CACHE['lxr_user_item']
        rcmd_list = DATA_CACHE['rcmd_list'].copy()
        base_list = DATA_CACHE['base_list']
        model = DATA_CACHE['model']
        user_embeds = DATA_CACHE['user_embeds']
        title_map = DATA_CACHE['title_map']
        
        # Case 1: No reviews - use trend-based filtering
        if len(user_reviews) == 0:
            current_date = datetime.now()
            result_df = recommend_trend_based(
                meta_sim, 
                lxr_review, 
                rcmd_list, 
                date=current_date, 
                month=month, 
                k=k
            )
            
            # Add titles
            result_df['title'] = result_df['asin'].map(title_map)
            
            recommendations = result_df[['title', 'asin', 'score']].to_dict('records')
            
            return jsonify({
                'recommendations': recommendations,
                'method': 'trend'
            })
        
        # Case 2: Has reviews - use hybrid approach with reranking
        else:
            # Extract purchased items and review texts
            purchased_asins = [review['asin'] for review in user_reviews]
            review_texts = [review['reviewText'] for review in user_reviews]
            combined_text = ' '.join(review_texts)
            
            # Remove purchased items from recommendation candidates
            rcmd_list = [item for item in rcmd_list if item not in purchased_asins]
            
            # 1. Content-based recommendations
            content_df = get_similarity_recommendations(
                meta_sim, 
                purchased_asins, 
                rcmd_list, 
                k=k*2,  # Get more candidates for reranking
                weight=None
            )
            
            # 2. Collaborative filtering recommendations
            target_emb = model.encode([combined_text], normalize_embeddings=True)
            collab_df = get_similar_users_from_reviews(
                lxr_user_item,
                target_emb,
                user_embeds,
                base_list,
                rcmd_list,
                meta_sim,
                k=k*2,  # Get more candidates for reranking
                m=m
            )
            
            # 3. Trend-based recommendations
            current_date = datetime.now()
            trend_df = recommend_trend_based(
                meta_sim,
                lxr_review,
                rcmd_list,
                date=current_date,
                month=month,
                k=k*2  # Get more candidates for reranking
            )
            
            # 4. Rerank recommendations
            result_df = rerank_recommendations(
                content_df,
                collab_df,
                trend_df,
                k=k,
                content_weight=0.4,
                collab_weight=0.4,
                trend_weight=0.2
            )
            
            # Add titles
            result_df['title'] = result_df['asin'].map(title_map)
            
            recommendations = result_df[['title', 'asin', 'score']].to_dict('records')
            
            return jsonify({
                'recommendations': recommendations,
                'method': 'hybrid'
            })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': len(DATA_CACHE) > 0
    })


if __name__ == '__main__':
    # Initialize data before starting the server
    initialize_data()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)