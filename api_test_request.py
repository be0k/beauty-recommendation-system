import requests

# 리뷰 없는 경우
response = requests.post(
    'http://localhost:5000/recommend',
    json={
        'user_reviews': [],
        'k': 10
    }
)
print(response.json())

# 리뷰 있는 경우
response = requests.post(
    'http://localhost:5000/recommend',
    json={
        'user_reviews': [
            {
                'asin': 'B000FTYALG',
                'reviewText': 'Great product!'
            }
        ],
        'k': 10
    }
)
print(response.json())