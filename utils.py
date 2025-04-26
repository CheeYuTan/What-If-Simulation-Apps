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
from typing import Generator, Optional, Dict, Any, Tuple
from requests.exceptions import HTTPError, RequestException
from datetime import datetime, timedelta

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
            return True
            
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
    """
    # Get warehouse ID from environment variable
    warehouse_id = os.getenv('DATABRICKS_WAREHOUSE_ID')
    if not warehouse_id:
        raise ValueError("DATABRICKS_WAREHOUSE_ID environment variable is not set")
    
    cfg = Config()
    with sql.connect(
        server_hostname=os.getenv('DATABRICKS_HOST'),
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
        staging_allowed_local_path="/tmp"  # Required for file ingestion commands
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()

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