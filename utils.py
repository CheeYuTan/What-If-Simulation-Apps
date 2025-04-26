import os
from databricks import sql
from databricks.sdk.core import Config
import pandas as pd
import numpy as np
import requests
import json
import base64
from dash import html
import logging
import time
from typing import Generator, Optional, Dict, Any, Tuple, Union
from requests.exceptions import HTTPError, RequestException
from datetime import datetime, timedelta
import io
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False

    def record_failure(self):
        current_time = datetime.now()
        if self.last_failure_time:
            # Reset failures if enough time has passed
            if current_time - self.last_failure_time > timedelta(seconds=self.reset_timeout):
                self.failures = 0
                self.is_open = False
        
        self.failures += 1
        self.last_failure_time = current_time
        
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
            
        # Check if enough time has passed to try again
        if self.last_failure_time and \
           datetime.now() - self.last_failure_time > timedelta(seconds=self.reset_timeout):
            self.is_open = False
            self.failures = 0
            
        return False

    def reset(self):
        self.failures = 0
        self.last_failure_time = None
        self.is_open = False

# Global circuit breaker instance
_circuit_breaker = CircuitBreaker()

def get_access_token() -> str:
    """
    Gets an access token using client credentials.
    
    Returns:
        str: The access token
    
    Raises:
        Exception: If unable to get access token
    """
    client_id = os.getenv('DATABRICKS_CLIENT_ID')
    client_secret = os.getenv('DATABRICKS_CLIENT_SECRET')
    host = os.getenv('DATABRICKS_HOST')
    
    if not all([client_id, client_secret, host]):
        raise ValueError("Required environment variables are not set")
    
    # Create basic auth header
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    
    # Set up the request
    headers = {
        'Authorization': f'Basic {auth}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials',
        'scope': 'all-apis'
    }
    
    try:
        response = requests.post(
            f"https://{host}/oidc/v1/token",
            headers=headers,
            data=data,
            timeout=30  # 30 seconds timeout for token request
        )
        response.raise_for_status()
        return response.json()['access_token']
    except Exception as e:
        print(f"Error getting access token: {str(e)}")
        raise

def sqlQuery(query: str) -> pd.DataFrame:
    """
    Executes a query against the Databricks SQL Warehouse and returns the result as a Pandas DataFrame.
    For PUT commands, handles the response as a file-like object.
    """
    cfg = Config()
    host = os.getenv('DATABRICKS_HOST')
    if not host:
        raise ValueError("DATABRICKS_HOST environment variable is not set")
        
    with sql.connect(
        server_hostname=host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate,
        staging_allowed_local_path="/tmp"  # Required for file ingestion commands
    ) as connection:
        with connection.cursor() as cursor:
            if query.strip().upper().startswith('PUT'):
                # For PUT commands, just execute without trying to fetch response
                print("Executing PUT command")
                try:
                    # Execute the PUT command
                    cursor.execute(query)
                    # PUT commands don't return data, so return empty DataFrame
                    return pd.DataFrame()
                except Exception as e:
                    # If the file is already in the volume, we can ignore the error
                    if "the JSON object must be str, bytes or bytearray, not list" in str(e):
                        print("Debug - PUT command completed (ignoring response processing error)")
                        return pd.DataFrame()
                    print(f"PUT command execution error: {str(e)}")
                    raise
            elif query.strip().upper().startswith('LIST'):
                # For LIST commands, execute and fetch results
                print("Executing LIST command")
                cursor.execute(query)
                # Read the results properly
                result = cursor.fetchall_arrow()
                if result:
                    return result.to_pandas()
                return pd.DataFrame()
            else:
                # For other queries, execute and fetch results
                cursor.execute(query)
                # Read the results properly
                result = cursor.fetchall_arrow()
                if result:
                    return result.to_pandas()
                return pd.DataFrame()

def read_table_data():
    """Read data from the configured Unity Catalog table."""
    try:
        # Get table path from environment variable
        table_path = os.getenv('SAMPLE_TABLE_PATH')
        if not table_path:
            raise ValueError("SAMPLE_TABLE_PATH environment variable is not set")
        
        # Split the table path into catalog, schema, and table
        catalog, schema, table = table_path.split('.')
        
        # Create a query to read the table
        query = f"SELECT * FROM {catalog}.{schema}.{table} LIMIT 1000"
        
        # Execute the query and get results
        df = sqlQuery(query)
        return df
            
    except Exception as e:
        print(f"Error reading table data: {str(e)}")
        raise

def get_table_schema():
    """Get the schema of the configured table."""
    try:
        # Get table path from environment variable
        table_path = os.getenv('SAMPLE_TABLE_PATH')
        if not table_path:
            raise ValueError("SAMPLE_TABLE_PATH environment variable is not set")
        
        # Split the table path into catalog, schema, and table
        catalog, schema, table = table_path.split('.')
        
        # Create a query to get schema
        query = f"DESCRIBE TABLE {catalog}.{schema}.{table}"
        
        # Execute the query and get results
        schema_df = sqlQuery(query)
        return schema_df
            
    except Exception as e:
        print(f"Error getting table schema: {str(e)}")
        raise

def get_column_stats():
    """Get statistics for each column in the table."""
    try:
        df = read_table_data()
        stats = {}
        
        for column in df.columns:
            col_type = df[column].dtype
            if np.issubdtype(col_type, np.number):
                # For numerical columns
                min_val = df[column].min()
                max_val = df[column].max()
                range_val = max_val - min_val
                stats[column] = {
                    'type': 'numeric',
                    'min': min_val - 0.25 * range_val,
                    'max': max_val + 0.25 * range_val,
                    'current_min': min_val,
                    'current_max': max_val
                }
            else:
                # For categorical columns
                unique_values = sorted(df[column].unique())
                stats[column] = {
                    'type': 'categorical',
                    'values': unique_values
                }
        
        return stats
    except Exception as e:
        print(f"Error getting column stats: {str(e)}")
        raise

def send_endpoint_request(data_dict: dict) -> dict:
    """
    Sends a request to the Databricks serving endpoint and returns the response.
    
    Args:
        data_dict (dict): Dictionary containing the input data in the format:
            {
                'fixed_acidity': float,
                'volatile_acidity': float,
                'citric_acid': float,
                'residual_sugar': float,
                'chlorides': float,
                'free_sulfur_dioxide': float,
                'total_sulfur_dioxide': float,
                'density': float,
                'pH': float,
                'sulphates': float,
                'alcohol': float,
                'is_red': int
            }
    
    Returns:
        dict: The response from the endpoint containing predictions and optional SHAP values
    
    Raises:
        ValueError: If required environment variables are not set
        Exception: If the request to the endpoint fails
    """
    # Get required environment variables
    endpoint_name = os.getenv('DATABRICKS_SERVING_ENDPOINT')
    host = os.getenv('DATABRICKS_HOST')
    
    if not endpoint_name or not host:
        raise ValueError("Required environment variables are not set")
    
    try:
        # Get access token using client credentials
        access_token = get_access_token()
        
        # Prepare the request
        url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "dataframe_split": {
                "columns": list(data_dict.keys()),
                "data": [list(data_dict.values())]
            }
        }
        
        # Log the request payload
        print(f"Sending request with payload: {payload}")
        
        # Send the request with a longer timeout for cold starts
        print("Sending request to endpoint (this may take a while if the endpoint needs to scale from zero)...")
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=300  # 5 minutes timeout for endpoint request
        )
        response.raise_for_status()
        
        # Parse and log the raw response
        raw_response = response.json()
        print(f"Raw response from endpoint: {raw_response}")
        print(f"Response type: {type(raw_response)}")
        
        # Return the raw response - we'll handle the nested structure in the app
        return raw_response
        
    except requests.exceptions.Timeout:
        print("Request timed out. The endpoint might be scaling up from zero, please try again.")
        raise
    except Exception as e:
        print(f"Error sending request to endpoint: {str(e)}")
        raise

def format_llm_prompt(prediction: float, input_features: dict, shap_values: list, feature_names: list) -> str:
    """
    Format prediction results into a prompt for the LLM interpreter.
    
    Args:
        prediction: The model's prediction value
        input_features: Dictionary of feature names and their values
        shap_values: List of SHAP values for each feature
        feature_names: List of feature names corresponding to SHAP values
    
    Returns:
        str: Formatted prompt for the LLM interpreter
    """
    # Sort features by absolute SHAP value
    feature_impacts = list(zip(feature_names, shap_values, [input_features[f] for f in feature_names]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Build the prompt
    prompt = f"""Analyze this wine quality prediction:

Predicted Quality Score: {prediction:.2f}

Input Features and Their Impacts (ordered by importance):"""
    
    # Add each feature's impact
    for feature, shap_value, value in feature_impacts:
        impact = "increased" if shap_value > 0 else "decreased"
        prompt += f"\n- {feature}: value = {value:.3f}, SHAP value = {shap_value:.3f} ({impact} the prediction)"
    
    prompt += "\n\nPlease provide a detailed interpretation following the format specified in your instructions."
    
    return prompt

def create_display_content(prediction: float, input_features: dict, shap_values: list, feature_names: list):
    """
    Create both visualization content and LLM prompt for display.
    
    Args:
        prediction: The model's prediction value
        input_features: Dictionary of feature names and their values
        shap_values: List of SHAP values for each feature
        feature_names: List of feature names corresponding to SHAP values
    
    Returns:
        list: List of Dash HTML components for display
    """
    # Sort features by absolute SHAP value
    feature_impacts = list(zip(feature_names, shap_values, [input_features[f] for f in feature_names]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Create alert content for display
    alert_content = [
        html.H4("Prediction Result", className="alert-heading"),
        html.Hr(),
        html.P(f"Quality Score: {float(prediction):.2f}", className="mb-0")
    ]
    
    # Add SHAP values to display
    shap_list = []
    for feature, shap_value, value in feature_impacts:
        shap_list.append(
            html.P(f"{feature}: {shap_value:.3f}", 
                  className="mb-1",
                  style={'color': 'red' if shap_value < 0 else 'green'})
        )
    alert_content.extend([
        html.Hr(),
        html.H5("Feature Contributions (SHAP Values):", className="mt-3"),
        *shap_list
    ])
    
    # Add formatted LLM prompt below
    llm_prompt = format_llm_prompt(prediction, input_features, shap_values, feature_names)
    alert_content.extend([
        html.Hr(),
        html.H5("Formatted LLM Prompt:", className="mt-3"),
        html.Pre(llm_prompt, style={
            'background-color': '#f8f9fa',
            'padding': '1rem',
            'border-radius': '0.25rem',
            'white-space': 'pre-wrap'
        })
    ])
    
    return alert_content 

def send_interpreter_request(prompt: str) -> str:
    """
    Send a prompt to the LLM interpreter endpoint and get interpretation.
    
    Args:
        prompt: The formatted prompt string containing prediction and SHAP values
    
    Returns:
        str: The interpretation text
    """
    try:
        # Get required environment variables
        endpoint_name = os.getenv('INTERPRETER_SERVING_ENDPOINT')
        if not endpoint_name:
            raise ValueError("INTERPRETER_SERVING_ENDPOINT environment variable is not set")
            
        host = os.getenv('DATABRICKS_HOST')
        if not host:
            raise ValueError("DATABRICKS_HOST environment variable is not set")
            
        # Get access token using client credentials
        access_token = get_access_token()
        
        # Prepare the request
        url = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        logger.info(f"Sending interpretation request to endpoint: {endpoint_name}")
        
        # Send the request
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        if isinstance(result, dict) and 'messages' in result and len(result['messages']) > 0:
            return result['messages'][0].get('content', '')
        
        return "No interpretation available"
            
    except Exception as e:
        logger.error(f"Error getting interpretation: {str(e)}", exc_info=True)
        raise

def stream_prediction_results(prediction: float, input_features: dict, shap_values: list, feature_names: list) -> Generator[Dict[str, Any], None, None]:
    """
    Stream prediction results one step at a time.
    
    Args:
        prediction: The model's prediction value
        input_features: Dictionary of feature names and their values
        shap_values: List of SHAP values for each feature
        feature_names: List of feature names corresponding to SHAP values
    
    Yields:
        Dict[str, Any]: Dictionary containing the current step's display information
    """
    try:
        # Step 1: Yield prediction
        yield {
            'type': 'prediction',
            'content': f"Predicted Quality Score: {prediction:.2f}"
        }
        
        # Step 2: Sort features by absolute SHAP value
        feature_impacts = list(zip(feature_names, shap_values, [input_features[f] for f in feature_names]))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Step 3: Yield each feature's impact
        for feature, shap_value, value in feature_impacts:
            impact = "increased" if shap_value > 0 else "decreased"
            yield {
                'type': 'feature_impact',
                'content': {
                    'feature': feature,
                    'value': value,
                    'shap_value': shap_value,
                    'impact': impact
                }
            }
            
    except Exception as e:
        logger.error(f"Error streaming prediction results: {str(e)}", exc_info=True)
        yield {
            'type': 'error',
            'content': f"Error: {str(e)}"
        }

def create_prediction_display(step_data: Dict[str, Any]) -> html.Div:
    """
    Create HTML display for a prediction step.
    
    Args:
        step_data: Dictionary containing the step information
        
    Returns:
        html.Div: Formatted display component
    """
    try:
        if step_data['type'] == 'prediction':
            return html.Div([
                html.H4("Prediction Result", className="mb-3"),
                html.P(step_data['content'], className="lead")
            ])
            
        elif step_data['type'] == 'feature_impact':
            impact = step_data['content']
            return html.Div([
                html.P([
                    html.Strong(f"{impact['feature']}: "),
                    f"value = {impact['value']:.3f}, impact = {impact['shap_value']:.3f}",
                    html.Span(
                        f" ({impact['impact']} prediction)",
                        style={'color': 'green' if impact['shap_value'] > 0 else 'red'}
                    )
                ], className="mb-2")
            ])
            
        elif step_data['type'] == 'error':
            return html.Div([
                html.P(step_data['content'], className="text-danger")
            ])
            
        return html.Div()  # Empty div for unknown types
        
    except Exception as e:
        logger.error(f"Error creating display: {str(e)}", exc_info=True)
        return html.Div([
            html.P(f"Error creating display: {str(e)}", className="text-danger")
        ]) 

def save_file_to_volume(encoded_content: str, volume_path: str, file_name: str, overwrite: bool = True) -> Tuple[str, str]:
    """
    Saves an uploaded file to a Databricks volume using the PUT command.

    Args:
        encoded_content (str): Base64 encoded file content.
        volume_path (str): The target Databricks volume path (e.g., "/Volumes/steventan/what_if_simulation_apps/batch_inference_upload").
        file_name (str): The name of the file to be saved.
        overwrite (bool): Whether to overwrite the existing file.

    Returns:
        Tuple[str, str]: (saved_path, sample_content)
            - saved_path: The full path of the saved file
            - sample_content: The first 5 lines of the file content as a string
    """
    try:
        print(f"\n=== Starting file upload process ===")
        print(f"Debug - Volume path received: {volume_path}")
        print(f"Debug - File name: {file_name}")
        
        # Check file size (limit to 100MB for example)
        content_string = encoded_content.split(",")[1]
        decoded = base64.b64decode(content_string)
        file_size = len(decoded)
        max_size = 100 * 1024 * 1024  # 100MB
        print(f"Debug - File size: {file_size} bytes")

        if file_size > max_size:
            raise ValueError(f"File size exceeds maximum limit of {max_size/1024/1024}MB")

        # Save to a temporary local file
        local_temp_path = f"/tmp/{file_name}"
        print(f"Debug - Local temp path: {local_temp_path}")

        # Check if /tmp directory exists and is writable
        if not os.path.exists('/tmp'):
            print("Debug - Creating /tmp directory")
            os.makedirs('/tmp')
        print(f"Debug - /tmp directory permissions: {oct(os.stat('/tmp').st_mode)[-3:]}")
            
        # Write the file
        print("Debug - Writing file to /tmp")
        with open(local_temp_path, "wb") as f:
            f.write(decoded)
            
        # Verify file was written
        if os.path.exists(local_temp_path):
            print(f"Debug - File written successfully. Size: {os.path.getsize(local_temp_path)} bytes")
            print("Debug - First few lines of file content:")
            with open(local_temp_path, 'r') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Only read first 5 lines
                        print(f"Line {i+1}: {line.strip()}")
                    else:
                        break
        else:
            print("Debug - ERROR: File was not written to /tmp")
            
        # Read first 5 lines for validation
        sample_lines = []
        with open(local_temp_path, 'r') as f:
            for i, line in enumerate(f):
                if i < 5:  # Only read first 5 lines
                    sample_lines.append(line)
                else:
                    break
        sample_content = ''.join(sample_lines)

        # Clean and normalize the volume path
        # Remove dbfs: prefix if present
        clean_path = volume_path
        if clean_path.startswith('dbfs:'):
            clean_path = clean_path[5:]
        # Ensure path starts with /Volumes
        if not clean_path.startswith('/Volumes'):
            clean_path = f"/Volumes{clean_path}"
            
        databricks_file_path = f"{clean_path}/{file_name}"
        overwrite_option = "OVERWRITE" if overwrite else ""
        print(f"Debug - Target path in volume: {databricks_file_path}")
        print(f"Debug - Overwrite option: {overwrite_option}")

        # Execute the Databricks SQL command to upload file
        query = f"PUT '{local_temp_path}' INTO '{databricks_file_path}' {overwrite_option}"
        print(f"Debug - SQL Query to execute: {query}")
        
        # List contents of /tmp before PUT
        print("\nDebug - Contents of /tmp before PUT command:")
        try:
            for file in os.listdir('/tmp'):
                print(f"- {file}")
        except Exception as e:
            print(f"Debug - Error listing /tmp contents: {str(e)}")
        
        # Execute the PUT command
        print("\nDebug - Executing PUT command...")
        try:
            sqlQuery(query)
            print("Debug - PUT command executed successfully")
        except Exception as e:
            print(f"Debug - PUT command error: {str(e)}")
            print(f"Debug - Error type: {type(e)}")
            raise
            
        # List contents of /tmp after PUT
        print("\nDebug - Contents of /tmp after PUT command:")
        try:
            for file in os.listdir('/tmp'):
                print(f"- {file}")
        except Exception as e:
            print(f"Debug - Error listing /tmp contents: {str(e)}")

        print(f"\nDebug - File successfully uploaded to: {databricks_file_path}")
        os.remove(local_temp_path)  # Cleanup local temp file
        
        return databricks_file_path, sample_content

    except Exception as e:
        print(f"\nDebug - Error uploading file to volume: {str(e)}")
        print(f"Debug - Error type: {type(e)}")
        raise

def validate_uploaded_file(file_path_or_content: Union[str, str]) -> Tuple[bool, str]:
    """
    Validates the uploaded file's structure and data types using the first 5 lines.
    
    Args:
        file_path_or_content (Union[str, str]): Either the path to the uploaded file in the Databricks volume
                                               or the first 5 lines of the file content as a string
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
            - is_valid: True if file is valid, False otherwise
            - error_message: Description of any validation errors
    """
    try:
        # Expected columns and their data types
        expected_columns = {
            'fixed_acidity': float,
            'volatile_acidity': float,
            'citric_acid': float,
            'residual_sugar': float,
            'chlorides': float,
            'free_sulfur_dioxide': float,
            'total_sulfur_dioxide': float,
            'density': float,
            'pH': float,
            'sulphates': float,
            'alcohol': float,
            'is_red': int
        }
        
        # Read the file content
        if isinstance(file_path_or_content, str) and file_path_or_content.startswith('/Volumes'):
            # Create a temporary local file to download the content
            local_temp_path = f"/tmp/validation_{os.path.basename(file_path_or_content)}"
            
            # Download the file from volume for validation
            query = f"GET '{file_path_or_content}' TO '{local_temp_path}'"
            print(f"Debug - Downloading file with query: {query}")
            
            # Execute the GET command
            cfg = Config()
            host = os.getenv('DATABRICKS_HOST')
            if not host:
                raise ValueError("DATABRICKS_HOST environment variable is not set")
                
            with sql.connect(
                server_hostname=host,
                http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
                credentials_provider=lambda: cfg.authenticate,
                staging_allowed_local_path="/tmp"
            ) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    print("Debug - File downloaded successfully")
            
            try:
                # Read first 5 lines
                sample_lines = []
                with open(local_temp_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i < 5:  # Only read first 5 lines
                            sample_lines.append(line)
                        else:
                            break
                sample_content = ''.join(sample_lines)
                df = pd.read_csv(io.StringIO(sample_content))
            finally:
                # Clean up the temporary file
                if os.path.exists(local_temp_path):
                    os.remove(local_temp_path)
                    print("Debug - Temporary file cleaned up")
        else:
            # Use the provided content directly
            print("Debug - Using provided file content")
            df = pd.read_csv(io.StringIO(file_path_or_content))
        
        print(f"Debug - File read successfully. Shape: {df.shape}")
        
        # Check number of columns
        if len(df.columns) != len(expected_columns):
            return False, f"Invalid number of columns. Expected {len(expected_columns)}, got {len(df.columns)}"
        
        # Check column names
        missing_columns = set(expected_columns.keys()) - set(df.columns)
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check data types
        type_errors = []
        for col, expected_type in expected_columns.items():
            try:
                # Try to convert column to expected type
                if expected_type == float:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                elif expected_type == int:
                    df[col] = df[col].astype(int)
            except Exception as e:
                type_errors.append(f"Column '{col}' should be {expected_type.__name__}: {str(e)}")
        
        if type_errors:
            return False, "\n".join(type_errors)
        
        # Check for missing values in the sample
        if df.isnull().any().any():
            return False, "File contains missing values"
        
        # Check value ranges in the sample
        range_errors = []
        for col in df.columns:
            if col == 'is_red':
                if not df[col].isin([0, 1]).all():
                    range_errors.append(f"Column '{col}' should only contain 0 or 1")
            else:
                if df[col].min() < 0:
                    range_errors.append(f"Column '{col}' contains negative values")
        
        if range_errors:
            return False, "\n".join(range_errors)
        
        return True, "File validation successful"
        
    except Exception as e:
        print(f"Debug - Error during validation: {str(e)}")
        return False, f"Error validating file: {str(e)}"

def run_batch_inference_job(file_name: str, folder_name: str) -> str:
    """
    Executes the batch inference job asynchronously with the specified parameters.
    
    Args:
        file_name (str): The name of the uploaded file to process
        folder_name (str): The user folder (email) where the file is stored
        
    Returns:
        str: The run ID of the job execution
    """
    try:
        # Get required environment variables
        job_id = os.getenv('DATABRICKS_BATCH_INFERENCE_JOB_ID')
        catalog = os.getenv('CATALOG')
        schema = os.getenv('SCHEMA')
        
        if not all([job_id, catalog, schema]):
            raise ValueError("Required environment variables are not set")
            
        # Get access token
        access_token = get_access_token()
        host = os.getenv('DATABRICKS_HOST')
        
        # Prepare the job parameters according to API spec
        job_params = {
            "job_id": int(job_id),
            "job_parameters": {
                "catalog_name": catalog,
                "schema_name": schema,
                "file_name": file_name,
                "folder_name": folder_name
            }
        }
        
        # Prepare the request with API version 2.2
        url = f"https://{host}/api/2.2/jobs/run-now"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        
        print(f"Starting batch inference job with parameters: {job_params}")
        
        # Execute the job
        response = requests.post(url, headers=headers, json=job_params)
        response.raise_for_status()
        
        run_id = response.json()['run_id']
        print(f"Job started successfully. Run ID: {run_id}")
        
        return run_id
        
    except Exception as e:
        print(f"Error starting batch inference job: {str(e)}")
        raise

def check_job_status(run_id: str) -> dict:
    """
    Checks the status of a job run and returns its details.
    Args:
        run_id (str): The run ID of the job execution
    Returns:
        dict: Job status information including:
            - state: The current state of the job
            - result_state: The final result state (if completed)
            - state_message: Any status message
            - job_name: The name of the job
            - creator: The user who started the job
            - start_time: Start time (ISO8601)
            - end_time: End time (ISO8601, if available)
            - cluster: Cluster info (id, name, type)
            - tasks: List of tasks (if available)
            - error: Any error info (if available)
    """
    try:
        # Get access token
        access_token = get_access_token()
        host = os.getenv('DATABRICKS_HOST')
        
        # Use the /api/2.2/jobs/runs/get endpoint for richer details
        url = f"https://{host}/api/2.2/jobs/runs/get"
        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }
        params = {
            "run_id": run_id
        }
        
        # Get run status
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        run_info = response.json()
        state = run_info['state']

        # Extract more details
        job_name = run_info.get('run_name')
        creator = run_info.get('creator_user_name')
        start_time = run_info.get('start_time')
        end_time = run_info.get('end_time')
        if start_time:
            from datetime import datetime
            start_time = datetime.utcfromtimestamp(start_time/1000).isoformat()
        if end_time:
            from datetime import datetime
            end_time = datetime.utcfromtimestamp(end_time/1000).isoformat()
        cluster = run_info.get('cluster_instance', {})
        tasks = run_info.get('tasks', [])
        error = run_info.get('error')

        result = {
            'state': state['life_cycle_state'],
            'result_state': state.get('result_state'),
            'state_message': state.get('state_message'),
            'job_name': job_name,
            'creator': creator,
            'start_time': start_time,
            'end_time': end_time,
            'cluster': cluster,
            'tasks': tasks,
            'error': error
        }
        return result
    except Exception as e:
        print(f"Error checking job status: {str(e)}")
        raise

def upload_file_to_volume_rest(encoded_content: str, volume_path: str, file_name: str, overwrite: bool = True) -> tuple:
    """
    Uploads a file to a Databricks volume using the DBFS REST API.

    Args:
        encoded_content (str): Base64 encoded file content.
        volume_path (str): The target Databricks volume path (e.g., "/Volumes/steventan/what_if_simulation_apps/batch_inference_upload").
        file_name (str): The name of the file to be saved.
        overwrite (bool): Whether to overwrite the existing file.

    Returns:
        tuple: (saved_path, sample_content)
            - saved_path: The full path of the saved file
            - sample_content: The first 5 lines of the file content as a string
    """
    import base64
    import os

    print(f"\n=== Starting REST API file upload process ===")
    print(f"Debug - Volume path received: {volume_path}")
    print(f"Debug - File name: {file_name}")

    # Prepare file content
    content_string = encoded_content.split(",")[1]
    decoded = base64.b64decode(content_string)
    file_size = len(decoded)
    max_size = 100 * 1024 * 1024  # 100MB
    print(f"Debug - File size: {file_size} bytes")
    if file_size > max_size:
        raise ValueError(f"File size exceeds maximum limit of {max_size/1024/1024}MB")

    # Save to a temporary local file for reading sample lines
    local_temp_path = f"/tmp/{file_name}"
    with open(local_temp_path, "wb") as f:
        f.write(decoded)
    sample_lines = []
    with open(local_temp_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                sample_lines.append(line)
            else:
                break
    sample_content = ''.join(sample_lines)

    # Clean and normalize the volume path
    clean_path = volume_path
    if clean_path.startswith('dbfs:'):
        clean_path = clean_path[5:]
    if not clean_path.startswith('/Volumes'):
        clean_path = f"/Volumes{clean_path}"
    dbfs_path = f"{clean_path}/{file_name}"

    # Prepare REST API call
    host = os.getenv('DATABRICKS_HOST')
    token = os.getenv('DATABRICKS_TOKEN') or os.getenv('DATABRICKS_BEARER_TOKEN')
    if not host or not token:
        raise ValueError("DATABRICKS_HOST and DATABRICKS_TOKEN environment variables must be set.")
    api_url = f"https://{host}/api/2.0/dbfs/put"
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "path": dbfs_path,
        "overwrite": overwrite,
        "contents": base64.b64encode(decoded).decode('utf-8')
    }
    print(f"Debug - Uploading to {dbfs_path} via REST API...")
    response = requests.post(api_url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Debug - REST API error: {response.status_code} {response.text}")
        raise Exception(f"Failed to upload file to DBFS: {response.text}")
    print(f"Debug - File successfully uploaded to: {dbfs_path}")
    os.remove(local_temp_path)
    return dbfs_path, sample_content 

def upload_file_to_volume_uc_files_api(encoded_content: str, volume_path: str, file_name: str, overwrite: bool = True, flask_request=None) -> tuple:
    """
    Uploads a file to a Unity Catalog volume using the Databricks SQL Connector for Python (PUT command).
    Optionally logs the X-Forwarded-Email header if flask_request is provided.

    Args:
        encoded_content (str): Base64 encoded file content.
        volume_path (str): The target Databricks volume path (e.g., "/Volumes/steventan/what_if_simulation_apps/batch_inference_upload").
        file_name (str): The name of the file to be saved.
        overwrite (bool): Whether to overwrite the existing file.
        flask_request: The Flask request object (optional)

    Returns:
        tuple: (saved_path, sample_content)
            - saved_path: The full path of the saved file
            - sample_content: The first 5 lines of the file content as a string
    """
    import base64
    import os
    import traceback
    from databricks import sql
    from utils import get_access_token
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("databricks-sql-connector").version
        print(f"databricks-sql-connector version: {version}")
    except Exception as e:
        print(f"Could not determine databricks-sql-connector version: {e}")

    # Print X-Forwarded-Email if available
    if flask_request is not None:
        email = flask_request.headers.get('X-Forwarded-Email')
        print(f"X-Forwarded-Email: {email}")

    print(f"\n=== Starting Databricks SQL Connector file upload process ===")
    print(f"Debug - Volume path received: {volume_path}")
    print(f"Debug - File name: {file_name}")

    # Prepare file content
    content_string = encoded_content.split(",")[1]
    decoded = base64.b64decode(content_string)
    file_size = len(decoded)
    max_size = 100 * 1024 * 1024  # 100MB
    print(f"Debug - File size: {file_size} bytes")
    if file_size > max_size:
        raise ValueError(f"File size exceeds maximum limit of {max_size/1024/1024}MB")

    # Save to a temporary local file for reading sample lines
    local_temp_path = f"/tmp/{file_name}"
    with open(local_temp_path, "wb") as f:
        f.write(decoded)
    sample_lines = []
    with open(local_temp_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                sample_lines.append(line)
            else:
                break
    sample_content = ''.join(sample_lines)

    # Clean and normalize the volume path
    clean_path = volume_path
    if clean_path.startswith('dbfs:'):
        clean_path = clean_path[5:]
    if not clean_path.startswith('/Volumes'):
        clean_path = f"/Volumes{clean_path}"
    file_path = f"{clean_path}/{file_name}"

    # Prepare SQL connection using Databricks Apps environment variables and OAuth flow
    host = os.getenv("DATABRICKS_HOST")
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    token = get_access_token()
    http_path = f"/sql/1.0/warehouses/{warehouse_id}" if warehouse_id else None

    print(f"Debug - DATABRICKS_HOST: {host}")
    print(f"Debug - DATABRICKS_WAREHOUSE_ID: {warehouse_id}")
    print(f"Debug - OAuth token (first 6 chars): {token[:6] if token else None}")
    print(f"Debug - HTTP Path: {http_path}")
    print(f"Debug - Local file exists: {os.path.exists(local_temp_path)}")
    print(f"Debug - Local file size: {os.path.getsize(local_temp_path) if os.path.exists(local_temp_path) else 'N/A'}")
    print(f"Debug - Target volume path: {file_path}")

    if not host or not warehouse_id or not token:
        raise ValueError("DATABRICKS_HOST, DATABRICKS_WAREHOUSE_ID, and OAuth credentials must be set.")

    put_query = f"PUT '{local_temp_path}' INTO '{file_path}' {'OVERWRITE' if overwrite else ''}"
    print(f"Debug - PUT query: {put_query}")

    try:
        print(f"Debug - Connecting to Databricks SQL Warehouse at {host} with warehouse ID {warehouse_id}")
        with sql.connect(
            server_hostname=host,
            http_path=http_path,
            access_token=token,
            staging_allowed_local_path="/tmp/"
        ) as connection:
            with connection.cursor() as cursor:
                print(f"Debug - Executing: {put_query}")
                cursor.execute(put_query)
                print(f"Debug - File successfully uploaded to: {file_path}")
    except Exception as e:
        print("Exception during PUT command:", str(e))
        traceback.print_exc()
        raise
    finally:
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)
    return file_path, sample_content 