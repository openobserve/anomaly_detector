from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pickle
import os
import rrcf
from typing import Dict, Any, List
from datetime import datetime

app = FastAPI(
    title="Anomaly Detection API",
    description="API for detecting anomalies using Random Cut Forest with time components",
    version="1.0.0"
)

class Point(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    minute: int = Field(..., ge=0, le=59, description="Minute of the hour (0-59)")
    y: float = Field(..., description="Main measurement value")

class AnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    anomaly_score_threshold: float
    time: Dict[str, int]
    value: float

# Load the trained model and threshold
def load_model():
    model_path = 'rrcf_model.pkl'
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    forest = []
    for tree_data in model_data['forest_data']:
        tree = rrcf.RCTree()
        for idx, point in enumerate(tree_data):
            tree.insert_point(point, index=idx)
        forest.append(tree)
    
    return forest, model_data['anomaly_score_threshold']

# Initialize the model and threshold
forest, ANOMALY_SCORE_THRESHOLD = load_model()

@app.get("/")
async def root():
    return {"message": "Welcome to Time Series Anomaly Detection API"}

@app.post("/detect_anomaly", response_model=AnomalyResponse)
async def detect_anomaly(point: Point) -> Dict[str, Any]:
    """
    Detect if a given point is an anomaly.
    
    Args:
        point: A point with time components and a measurement value
        
    Returns:
        Dictionary containing:
        - is_anomaly: Boolean indicating if the point is an anomaly
        - anomaly_score: The calculated anomaly score
        - anomaly_score_threshold: The threshold used for anomaly detection
        - time: Dictionary of time components
        - value: The input measurement value
    """
    try:
        # Convert point to numpy array with all dimensions
        point_array = np.array([[
            point.hour,
            point.minute,
            point.y
        ]])
        
        # Calculate anomaly score
        anomaly_scores = []
        
        for tree in forest:
            # Create a copy of the tree
            tree_copy = rrcf.RCTree()
            # Copy all points from the original tree
            for idx in tree.leaves:
                tree_copy.insert_point(tree.leaves[idx].x, index=idx)
            
            # Insert the new point
            new_idx = len(tree_copy.leaves)
            tree_copy.insert_point(point_array[0], index=new_idx)
            
            # Calculate codisp for the new point
            score = tree_copy.codisp(new_idx)
            if score is not None:
                anomaly_scores.append(score)
        
        # Calculate average anomaly score
        if anomaly_scores:
            anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
        else:
            raise HTTPException(status_code=500, detail="Failed to calculate anomaly score")
        
        # Determine if it's an anomaly based on threshold
        is_anomaly = anomaly_score > ANOMALY_SCORE_THRESHOLD
        
        # Create time dictionary
        time = {
            "hour": point.hour,
            "minute": point.minute
        }
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "anomaly_score_threshold": ANOMALY_SCORE_THRESHOLD,
            "time": time,
            "value": point.y
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def model_info():
    """Get information about the trained model."""
    return {
        "num_trees": len(forest),
        "tree_size": len(forest[0].leaves) if forest else 0,
        "threshold": ANOMALY_SCORE_THRESHOLD,
        "dimensions": [
            "hour",
            "minute",
            "value"
        ]
    }

def main():
    """Main function to start the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)

if __name__ == "__main__":
    main() 