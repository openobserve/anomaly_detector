import numpy as np
import rrcf
import matplotlib.pyplot as plt
import pickle
import duckdb
from datetime import datetime, UTC
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial
import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set dark mode style
plt.style.use('dark_background')

def getSQLQuery():
    """Returns the SQL query for fetching time series data."""
    return """
    SELECT 
        date_part('hour', to_timestamp(_timestamp / 1000000)) AS hour,
        date_part('minute', to_timestamp(_timestamp / 1000000)) AS minute,
        COUNT(*) AS y
    FROM "default"
    GROUP BY hour, minute
    ORDER BY hour, minute
    LIMIT 2000
    """

def get_data_from_o2_api(start_time, end_time, sql_query=None):
    """Fetch data from OpenObserve API using the provided SQL query and time range."""
    # API configuration
    base_url = os.getenv('ORIGIN_CLUSTER_URL')
    token = os.getenv('ORIGIN_CLUSTER_TOKEN')
    org = os.getenv('OPENOBSERVE_ORG')
    
    if not all([base_url, token, org]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    # Construct API endpoint
    endpoint = f"{base_url}api/{org}/_search"
    
    # Use a simple SQL query for debugging
    debug_sql_query = 'SELECT * FROM "default" LIMIT 1'
    payload = {
        "query": {
            "sql": sql_query if sql_query else debug_sql_query,
            "start_time": start_time,
            "end_time": end_time,
            "size": 2000,   # In 1 day we should receive 1440 rows max
            "sql_mode": "full",
            "track_total_hits": False
        }
    }
    
    # Set headers
    headers = {
        'Authorization': f'Basic {token}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Make API request
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Extract and format the results
        results = []
        if 'hits' in data:
            for row in data['hits']:
                if all(key in row for key in ['hour', 'minute', 'y']):
                    results.append([
                        float(row['hour']),
                        float(row['minute']),
                        float(row['y'])
                    ])
        
        if not results:
            print("Warning: No data points found in the API response")
            print("API Response:", json.dumps(data, indent=2))
            return np.array([], dtype=np.float64)
        
        return np.array(results, dtype=np.float64)
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from OpenObserve API: {e}")
        raise

def load_data_from_api():
    """Load and process data from OpenObserve API."""
    # Set end_time as NOW in Unix epoch microseconds using timezone-aware UTC
    end_time = int(datetime.now(UTC).timestamp() * 1_000_000)
    # Set start_time as one day ago in Unix epoch microseconds
    start_time = end_time - (24 * 60 * 60 * 1_000_000)  # 24 hours in microseconds
    
    # Get SQL query
    sql_query = getSQLQuery()
    
    # Fetch data from API
    X = get_data_from_o2_api(start_time, end_time, sql_query)
    
    if len(X) == 0:
        raise ValueError("No data points received from the API. Please check the API response and time range.")
    
    print(f"Loaded {len(X)} data points")
    return X

def calculate_anomaly_scores(X, forest):
    """Calculate anomaly scores for each point in the dataset using streaming approach."""
    print("Calculating anomaly scores (streaming method)...")
    anomaly_scores = np.zeros(X.shape[0])
    for i, point in tqdm(enumerate(X), total=len(X), desc="Scoring points"):
        point_scores = []
        for tree in forest:
            idx = len(tree.leaves)
            tree.insert_point(point, index=idx)
            score = tree.codisp(idx)
            tree.forget_point(idx)
            if score is not None:
                point_scores.append(score)
        if point_scores:
            anomaly_scores[i] = np.mean(point_scores)
    return anomaly_scores

def plot_anomaly_detection(X, anomaly_scores):
    """Create and save a visualization of the anomaly detection results."""
    plt.figure(figsize=(15, 8))

    # Plot y values over time
    time_points = np.arange(len(X))
    plt.plot(time_points, X[:, 2], 'cyan', alpha=0.5, label='Y Values')

    # Add anomaly scores as color gradient
    scatter = plt.scatter(time_points, X[:, 2], 
                         c=anomaly_scores, cmap='plasma', 
                         alpha=0.6, s=50)
    plt.colorbar(scatter, label='Anomaly Score')

    plt.title('Time Series Anomaly Detection', color='white')
    plt.xlabel('Time Points', color='white')
    plt.ylabel('Y Values', color='white')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.savefig('anomaly_detection.png', facecolor='black', edgecolor='none')
    plt.close()

def train_forest(X, num_trees=100):
    """Train a Random Cut Forest model on the input data."""
    print("\nTraining new model...")
    forest = []
    forest_data = []  # Store the data used to create each tree

    if len(X) == 0:
        raise ValueError("No data points available for training")

    # Adjust tree size for small datasets
    tree_size = min(256, max(10, len(X)))
    print(f"Using {tree_size} points for each tree")
    
    for _ in range(num_trees):
        # Sample points from the dataset
        if tree_size >= len(X):
            # If tree size is larger than dataset, use all points
            sample = X.copy()
        else:
            # Otherwise sample randomly
            sample_indices = np.random.choice(X.shape[0], size=tree_size, replace=False)
            sample = X[sample_indices]
        
        tree = rrcf.RCTree(sample)
        forest.append(tree)
        forest_data.append(sample.copy())

    print("Model trained successfully!")
    return forest, forest_data

def main():
    # Load data from API instead of CSV
    X = load_data_from_api()

    # Train the forest
    forest, forest_data = train_forest(X)

    # Calculate anomaly scores
    anomaly_scores = calculate_anomaly_scores(X, forest)

    # Calculate dynamic threshold using percentile
    THRESHOLD_PERCENTILE = 98  # Points above 98th percentile are considered anomalies
    anomaly_score_threshold = np.percentile(anomaly_scores, THRESHOLD_PERCENTILE)
    print(f"Calculated anomaly threshold (98th percentile): {anomaly_score_threshold:.2f}")

    # Save both the forest data and threshold
    model_data = {
        'forest_data': forest_data,
        'anomaly_score_threshold': float(anomaly_score_threshold)
    }
    with open('rrcf_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Model and threshold saved to rrcf_model.pkl!")

    # Create visualization
    plot_anomaly_detection(X, anomaly_scores)

if __name__ == '__main__':
    main()
