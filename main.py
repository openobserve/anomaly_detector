# This the main entry point for the Action
import os
import sys
import json
import logging
import requests
import base64
import time
import uuid
import numpy as np
import pickle
import rrcf
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from fastapi import HTTPException

# Load environment variables from .env file
load_dotenv()

# Constants
MODEL_PATH = 'rrcf_model.pkl'
DEFAULT_BASE_URL = "http://localhost:5080"
DEFAULT_TOKEN = "openobserve:openobserve"
DEFAULT_SQL_QUERY = """
    SELECT 
        date_part('hour', to_timestamp(_timestamp / 1000000)) AS "hour",
        date_part('minute', to_timestamp(_timestamp / 1000000)) AS "minute",
        COUNT(*) AS "y"
    FROM "default"
    GROUP BY hour, minute
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OpenObserveConfig:
    """Configuration for OpenObserve client."""
    base_url: str
    token: str
    org: str
    stream: str
    sql_query: str
    time_window_minutes: int
    page_size: int
    quick_mode: bool
    sql_mode: str

class Point(BaseModel):
    """Data point model for anomaly detection."""
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    minute: int = Field(..., ge=0, le=59, description="Minute of the hour (0-59)")
    y: float = Field(..., description="Main measurement value")

class AnomalyResponse(BaseModel):
    """Response model for anomaly detection results."""
    is_anomaly: bool
    anomaly_score: float
    anomaly_score_threshold: float
    time: Dict[str, int]
    value: float

class ModelManager:
    """Manages the RRCF model and its operations."""
    
    def __init__(self):
        self.forest: List[rrcf.RCTree] = []
        self.anomaly_score_threshold: float = 0.0
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained model and threshold from disk."""
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            self.forest = []
            for tree_data in model_data['forest_data']:
                tree = rrcf.RCTree()
                for idx, point in enumerate(tree_data):
                    tree.insert_point(point, index=idx)
                self.forest.append(tree)
            
            self.anomaly_score_threshold = model_data['anomaly_score_threshold']
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to load model")

    def detect_anomaly(self, point: Point) -> AnomalyResponse:
        """Detect anomaly in the given data point."""
        try:
            point_array = np.array([[point.hour, point.minute, point.y]])
            anomaly_scores = []
            
            for tree in self.forest:
                tree_copy = rrcf.RCTree()
                for idx in tree.leaves:
                    tree_copy.insert_point(tree.leaves[idx].x, index=idx)
                
                new_idx = len(tree_copy.leaves)
                tree_copy.insert_point(point_array[0], index=new_idx)
                
                score = tree_copy.codisp(new_idx)
                if score is not None:
                    anomaly_scores.append(score)
            
            if not anomaly_scores:
                raise ValueError("Failed to calculate anomaly score")
            
            anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
            is_anomaly = anomaly_score > self.anomaly_score_threshold
            
            return AnomalyResponse(
                is_anomaly=is_anomaly,
                anomaly_score=float(anomaly_score),
                anomaly_score_threshold=self.anomaly_score_threshold,
                time={"hour": point.hour, "minute": point.minute},
                value=point.y
            )
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            raise

class OpenObserveClient:
    """Client for interacting with OpenObserve API."""
    
    def __init__(self, config: OpenObserveConfig):
        self.config = config
        self._setup_auth()
        logger.info(f"Initialized OpenObserve client for org: {config.org}, stream: {config.stream}")

    def _setup_auth(self) -> None:
        """Setup authentication for OpenObserve API."""
        try:
            decoded = base64.b64decode(self.config.token).decode('utf-8')
            self.username, self.password = decoded.split(':', 1)
            self.auth = (self.username, self.password)
        except Exception as e:
            error_msg = f"Failed to parse base64 encoded token: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def store_data(self, data: Dict[str, Any]) -> bool:
        """Store data in OpenObserve stream."""
        try:
            url = f"{self.config.base_url}/api/{self.config.org}/{self.config.stream}/_json"
            response = requests.post(
                url,
                json=data,
                auth=self.auth,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in (200, 201, 204):
                logger.info(f"Successfully stored data in {self.config.stream}")
                return True
            
            logger.error(f"Failed to store data: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"Error while storing data: {str(e)}")
            return False

    def get_data(self) -> Optional[Dict[str, Any]]:
        """Get data from OpenObserve stream."""
        try:
            url = f"{self.config.base_url}/api/{self.config.org}/_search?type=logs&search_type=ui&use_cache=true"
            trace_id = str(uuid.uuid4())
            
            current_time = int(time.time())
            current_minute = (current_time // 60) * 60
            current_time_micro = current_minute * 1000000
            
            data = {
                "query": {
                    "sql": self.config.sql_query,
                    "start_time": current_time_micro - (self.config.time_window_minutes * 60 * 1000000),
                    "end_time": current_time_micro,
                    "from": 0,
                    "size": self.config.page_size,
                    "quick_mode": self.config.quick_mode,
                    "sql_mode": self.config.sql_mode
                }
            }

            response = requests.post(
                url,
                json=data,
                auth=self.auth,
                headers={
                    "Content-Type": "application/json",
                    "traceparent": f"00-{trace_id}-00"
                }
            )

            if response.status_code in (200, 201, 204):
                logger.info("Successfully retrieved data")
                return response.json()
            
            logger.error(f"Failed to get data: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error while getting data: {str(e)}")
            return None

def get_config() -> OpenObserveConfig:
    """Get configuration from environment variables."""
    return OpenObserveConfig(
        base_url=os.getenv("ORIGIN_CLUSTER_URL", DEFAULT_BASE_URL).rstrip('/'),
        token=os.getenv("ORIGIN_CLUSTER_TOKEN", DEFAULT_TOKEN),
        org=os.getenv("OPENOBSERVE_ORG", "default"),
        stream=os.getenv("OPENOBSERVE_STREAM", "volume_log_anomaly"),
        sql_query=os.getenv("OPENOBSERVE_SQL_QUERY", DEFAULT_SQL_QUERY),
        time_window_minutes=int(os.getenv("OPENOBSERVE_TIME_WINDOW_MINUTES", "1")),
        page_size=int(os.getenv("OPENOBSERVE_PAGE_SIZE", "2000")),
        quick_mode=os.getenv("OPENOBSERVE_QUICK_MODE", "false").lower() == "true",
        sql_mode=os.getenv("OPENOBSERVE_SQL_MODE", "full")
    )

def main() -> None:
    """Main entry point for the application."""
    try:
        config = get_config()
        client = OpenObserveClient(config)
        model_manager = ModelManager()
        
        logger.info("Getting data from OpenObserve")
        response_data = client.get_data()
        
        if not response_data or "hits" not in response_data or not response_data["hits"]:
            logger.error("No data received from OpenObserve")
            return

        hit_data = response_data["hits"][0]
        try:
            point = Point(
                hour=hit_data["hour"],
                minute=hit_data["minute"],
                y=hit_data["y"]
            )
        except ValidationError as e:
            logger.error(f"Invalid data format: {str(e)}")
            return

        anomaly_response = model_manager.detect_anomaly(point)
        logger.info(f"Anomaly detection result: {json.dumps(anomaly_response.model_dump(), indent=4)}")
        
        response_data = client.store_data(anomaly_response.model_dump())
        logger.info(f"Stored data: {json.dumps(response_data, indent=4)}")

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()
    
