import subprocess
import sys
import os
import zipfile
import requests
from dotenv import load_dotenv
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Load environment variables from .env file
load_dotenv()

def create_action():
    """
    Function to create and upload the action to OpenObserve
    """
    print("Creating the action")
    try:
        # Get environment variables
        org_id = os.getenv('OPENOBSERVE_ORG')
        api_url = os.getenv('ORIGIN_CLUSTER_URL')
        
        if not org_id or not api_url:
            raise ValueError("Required environment variables OPENOBSERVE_ORG and ORIGIN_CLUSTER_URL must be set in .env file")
        
        # Construct API endpoint
        url = f"{api_url}/api/{org_id}/actions/upload"
        print(f"Uploading to: {url}")
        
        # Prepare and validate zip file
        action_name = "anomaly_detector"
        zip_path = f"{action_name}.zip"
        
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file {zip_path} not found. Run pack() first.")
            
        # Validate zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if zip_ref.testzip() is not None:
                    raise ValueError("Zip file is corrupted")
                print("Zip file contents:", zip_ref.namelist())
        except zipfile.BadZipFile:
            raise ValueError("Invalid zip file format")
        
        # Set essential headers
        headers = {
            'accept': 'application/json, text/plain, */*',
            'referrer': f'{api_url}/web/actions?action=add&org_identifier={org_id}'
        }
        
        # Add authorization if token exists
        api_token = os.getenv('ORIGIN_CLUSTER_TOKEN')
        if api_token:
            headers['Authorization'] = f'Basic {api_token}'
        
        # Prepare multipart form data
        with open(zip_path, 'rb') as f:
            m = MultipartEncoder(
                fields=[
                    ('name', 'anomaly_detector'),
                    ('description', ''),
                    ('execution_details', 'repeat'),
                    ('service_account', 'anomaly_detector@openobserve.ai'),
                    ('cron_expr', '* * * * *'),
                    ('owner', ''),
                    ('environment_variables', '{}'),
                    ('file', (f'{action_name}.zip', f, 'application/zip')),
                    ('filename', f'{action_name}.zip')
                ]
            )
            headers['Content-Type'] = m.content_type
            
            # Make the request
            response = requests.post(
                url, 
                data=m, 
                headers=headers
            )
        
        # Check response
        if response.status_code == 200:
            print("Action uploaded successfully!")
            return True
        else:
            print(f"Failed to upload action. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error occurred while creating action: {str(e)}")
        return False

def train():
    """
    Function to run train.py script with real-time output
    """
    try:
        result = subprocess.run([sys.executable, 'train.py'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("Error occurred during training")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return False

def pack():
    """
    Function to pack the necessary files into a zip archive
    """
    print("Packing the action")
    try:
        action_name = "anomaly_detector"
        files_to_zip = ['pyproject.toml', 'rrcf_model.pkl', 'main.py']
        
        # Create zip file using subprocess to match shell script
        zip_cmd = ['zip', '-r', f'{action_name}.zip'] + files_to_zip
        result = subprocess.run(zip_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error creating zip: {result.stderr}")
            return False
            
        print("Successfully created zip file")
        return True
    except Exception as e:
        print(f"Error occurred while packing: {str(e)}")
        return False

if __name__ == "__main__":
    train()
    pack()
    create_action()
