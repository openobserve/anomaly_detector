import requests
from datetime import datetime
import json
from tabulate import tabulate
import random
import sys
from tqdm import tqdm

def generate_test_samples(num_samples=10):
    """Generate test samples with both normal and anomalous values."""
    now = datetime.now()
    samples = []
    
    # Generate normal samples (20-25)
    normal_samples = [
        {
            "hour": now.hour,
            "minute": 0,
            "y": random.uniform(150000, 200000)
        }
        for _ in range(num_samples // 2)
    ]
    
    # Generate anomalous samples (50-95)
    anomalous_samples = [
        {
            "hour": now.hour,
            "minute": 0,
            "y": random.uniform(400000, 1000000)
        }
        for _ in range(num_samples // 2)
    ]
    
    # Add edge cases. These are anomalies that are not in the training data
    edge_cases = [
        {
            "hour": now.hour,
            "minute": 0,
            "y": 0  # Minimum value
        },
        {
            "hour": now.hour,
            "minute": 0,
            "y": 100000000  # Maximum value
        }
    ]
    
    return normal_samples + anomalous_samples + edge_cases

def test_anomaly_detection(samples):
    """Test anomaly detection for given samples."""
    results = []
    
    for i, sample in enumerate(tqdm(samples, desc="Testing samples", unit="sample"), 1):
        try:
            response = requests.post(
                "http://localhost:7000/detect_anomaly",
                json=sample,
                timeout=5  # Add timeout
            )
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json()
            
            # Get anomaly score and threshold
            anomaly_score = result.get("anomaly_score", 0)
            anomaly_threshold = result.get("anomaly_score_threshold", 0)
            is_anomaly = result.get("is_anomaly", False)
            
            results.append([
                i,
                sample["y"],
                is_anomaly,
                anomaly_score,
                anomaly_threshold
            ])
            
        except requests.exceptions.RequestException as e:
            print(f"\nError testing sample {i}: {str(e)}")
            results.append([
                i,
                sample["y"],
                "Error",
                "Error",
                "Error"
            ])
    
    return results

def main():
    try:
        # Generate test samples
        test_samples = generate_test_samples(10)
        
        # Test all samples and collect results
        results = test_anomaly_detection(test_samples)
        
        # Print results in a table
        headers = [
            "Sample #", 
            "Y Value (Target)", 
            "Anomaly Detected", 
            "Anomaly Score", 
            "Anomaly Threshold"
        ]
        print("\nAnomaly Detection Test Results:")
        print(tabulate(results, headers=headers, tablefmt="grid", numalign="center", stralign="center"))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
