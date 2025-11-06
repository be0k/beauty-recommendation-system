from src.utils import getDF
import pandas as pd
import numpy as np

def preprocessing(review, meta, fill, DATA_DIR):
    meta = meta.replace(["[]", "{}", "", " ", "nan", "NaN", "None",], np.nan)
    meta['description'] = meta['description'].apply(lambda x: x[0] if len(x)!=0 else np.nan)
    #print(meta['description'])
    meta.to_csv(DATA_DIR / 'tmp.csv', index=False)
    meta = pd.read_csv(DATA_DIR / 'tmp.csv',
                    na_values=["[]", "{}", "", " ", "None", "nan"])
    
    meta = meta.drop_duplicates(subset='asin', keep='last')

    #서로 중복되는 것만
    meta = meta[meta['asin'].isin(review['asin'].unique())]
    meta = meta.reset_index(drop=True)

    review = review[review['asin'].isin(meta['asin'].unique())]
    review = review.reset_index(drop=True)
    
    #null값 채우기?
    for idx, text in fill.items():
        meta.loc[meta['asin']==idx, 'description'] = text
    
    #null값 제거
    null_asins = meta.loc[meta["description"].isnull(), "asin"].unique()
    review = review[~review["asin"].isin(null_asins)]
    meta = meta[~meta["description"].isnull()]
    review = review[~review["reviewText"].isnull()]

    #보기좋게
    meta = meta.reset_index(drop=True)
    review = review.reset_index(drop=True)

    review = review[['asin', 'overall', 'reviewerID', 'reviewTime','reviewText']]
    meta = meta[['description', 'title', 'asin']]

    return review, meta
    
