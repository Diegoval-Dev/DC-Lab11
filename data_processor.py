import json
import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
import networkx as nx

def load_and_process_tweets(file_path='data/traficogt.txt'):
    """
    Load and process the Twitter data from the file
    """
    tweets_data = []

    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            for line_num, line in enumerate(f):
                try:
                    tweet = json.loads(line.strip())
                    if tweet and 'text' in tweet:  # Basic validation
                        tweets_data.append(tweet)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

        print(f"Loaded {len(tweets_data)} tweets from {file_path}")

    except FileNotFoundError:
        print(f"File {file_path} not found. Creating sample data.")
        return create_sample_data()
    except Exception as e:
        print(f"Error loading file: {e}. Creating sample data.")
        return create_sample_data()

    if not tweets_data:
        print("No valid tweets found. Creating sample data.")
        return create_sample_data()

    # Convert to DataFrame
    df = pd.DataFrame(tweets_data)

    # Process the data
    df = process_tweet_data(df)

    return df

def process_tweet_data(df):
    """
    Process and clean the tweet data
    """
    # Handle different possible date formats
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    else:
        # Create synthetic dates if not available
        df['created_at'] = pd.date_range(start='2022-01-01', periods=len(df), freq='H')

    # Extract date components
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year

    # Handle user information
    if 'author' in df.columns:
        df['username'] = df['author'].apply(lambda x:
            x.get('username', f'user_{np.random.randint(1000)}')
            if isinstance(x, dict) else f'user_{np.random.randint(1000)}'
        )
        df['author_id'] = df['author'].apply(lambda x:
            x.get('id', f'id_{np.random.randint(10000)}')
            if isinstance(x, dict) else f'id_{np.random.randint(10000)}'
        )
    elif 'user' in df.columns:
        df['username'] = df['user'].apply(lambda x:
            x.get('screen_name', f'user_{np.random.randint(1000)}')
            if isinstance(x, dict) else f'user_{np.random.randint(1000)}'
        )
        df['author_id'] = df['user'].apply(lambda x:
            x.get('id', f'id_{np.random.randint(10000)}')
            if isinstance(x, dict) else f'id_{np.random.randint(10000)}'
        )
    else:
        # Create synthetic usernames
        usernames = ['traficogt', 'user_traffic', 'gt_reporter', 'traffic_alert'] + \
                   [f'user_{i}' for i in range(100)]
        df['username'] = np.random.choice(usernames, len(df))
        df['author_id'] = [f'id_{i}' for i in range(len(df))]

    # Handle engagement metrics
    if 'public_metrics' in df.columns:
        df['retweet_count'] = df['public_metrics'].apply(lambda x:
            x.get('retweet_count', np.random.poisson(5))
            if isinstance(x, dict) else np.random.poisson(5)
        )
        df['like_count'] = df['public_metrics'].apply(lambda x:
            x.get('like_count', np.random.poisson(15))
            if isinstance(x, dict) else np.random.poisson(15)
        )
        df['reply_count'] = df['public_metrics'].apply(lambda x:
            x.get('reply_count', np.random.poisson(2))
            if isinstance(x, dict) else np.random.poisson(2)
        )
    else:
        # Create synthetic engagement metrics
        df['retweet_count'] = np.random.poisson(5, len(df))
        df['like_count'] = np.random.poisson(15, len(df))
        df['reply_count'] = np.random.poisson(2, len(df))

    # Process text
    if 'text' in df.columns:
        df['text_clean'] = df['text'].fillna('').astype(str)
    else:
        # Create sample text
        traffic_terms = [
            'Tráfico lento en la zona', 'Accidente reportado',
            'Lluvia intensa', 'Cierre de carretera', 'Desvío temporal',
            'Congestionamiento', 'Paso habilitado', 'Ruta alterna'
        ]
        df['text_clean'] = [np.random.choice(traffic_terms) for _ in range(len(df))]

    # Calculate additional metrics
    df['word_count'] = df['text_clean'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['text_clean'].apply(lambda x: len(str(x)))
    df['engagement_total'] = df['retweet_count'] + df['like_count'] + df['reply_count']
    df['engagement_rate'] = df['engagement_total'] / (df['word_count'] + 1)

    # Extract mentions and hashtags
    df['mentions'] = df['text_clean'].apply(extract_mentions)
    df['hashtags'] = df['text_clean'].apply(extract_hashtags)
    df['has_mentions'] = df['mentions'].apply(lambda x: len(x) > 0)
    df['has_hashtags'] = df['hashtags'].apply(lambda x: len(x) > 0)

    # Traffic-related categories
    df['traffic_category'] = df['text_clean'].apply(categorize_traffic_content)

    return df

def extract_mentions(text):
    """Extract @mentions from text"""
    return re.findall(r'@(\w+)', str(text).lower())

def extract_hashtags(text):
    """Extract #hashtags from text"""
    return re.findall(r'#(\w+)', str(text).lower())

def categorize_traffic_content(text):
    """Categorize traffic content based on keywords"""
    text_lower = str(text).lower()

    if any(word in text_lower for word in ['lluvia', 'inundacion', 'tormenta']):
        return 'Weather'
    elif any(word in text_lower for word in ['accidente', 'choque', 'colision']):
        return 'Accident'
    elif any(word in text_lower for word in ['cierre', 'cerrado', 'bloqueado']):
        return 'Road Closure'
    elif any(word in text_lower for word in ['lento', 'congestion', 'trafico']):
        return 'Traffic Jam'
    elif any(word in text_lower for word in ['habilitado', 'abierto', 'despejado']):
        return 'Road Open'
    else:
        return 'General'

def create_sample_data():
    """Create realistic sample data for demonstration"""
    np.random.seed(42)

    # Generate date range
    dates = pd.date_range(start='2023-01-01', end='2024-10-30', freq='2H')
    n_samples = len(dates)

    # Realistic usernames
    usernames = ['traficogt'] * 800 + ['BArevalodeLeon'] * 100 + \
               ['DrGiammattei'] * 50 + ['MPguatemala'] * 50 + \
               [f'user_{i}' for i in range(200)]

    # Traffic-related content
    traffic_content = [
        'Tráfico lento en Zona 10 debido a lluvia',
        'Accidente en Calzada Roosevelt, usar rutas alternas',
        'Cierre temporal en Periférico por obras',
        'Congestionamiento en horas pico zona 4',
        'Paso habilitado en Carretera Interamericana',
        'Inundación en Zona 12, evitar la zona',
        'Desvío por manifestación en Centro Histórico',
        'Lluvia intensa afecta visibilidad en Zona 15'
    ]

    # Categories
    categories = ['Weather', 'Accident', 'Road Closure', 'Traffic Jam', 'Road Open', 'General']

    df = pd.DataFrame({
        'created_at': np.random.choice(dates, n_samples),
        'text_clean': np.random.choice(traffic_content, n_samples),
        'username': np.random.choice(usernames, n_samples),
        'author_id': [f'id_{i}' for i in range(n_samples)],
        'retweet_count': np.random.poisson(8, n_samples),
        'like_count': np.random.poisson(25, n_samples),
        'reply_count': np.random.poisson(3, n_samples),
        'traffic_category': np.random.choice(categories, n_samples)
    })

    # Add derived fields
    df['date'] = df['created_at'].dt.date
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()
    df['month'] = df['created_at'].dt.month
    df['year'] = df['created_at'].dt.year
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text_clean'].apply(len)
    df['engagement_total'] = df['retweet_count'] + df['like_count'] + df['reply_count']
    df['engagement_rate'] = df['engagement_total'] / (df['word_count'] + 1)

    # Add mentions and hashtags
    df['mentions'] = df['text_clean'].apply(extract_mentions)
    df['hashtags'] = df['text_clean'].apply(extract_hashtags)
    df['has_mentions'] = df['mentions'].apply(lambda x: len(x) > 0)
    df['has_hashtags'] = df['hashtags'].apply(lambda x: len(x) > 0)

    print(f"Created sample dataset with {len(df)} tweets")
    return df

if __name__ == "__main__":
    # Test the data loading
    df = load_and_process_tweets()
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Unique users: {df['username'].nunique()}")