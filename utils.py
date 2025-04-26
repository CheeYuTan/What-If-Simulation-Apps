import os
from databricks import sql
from databricks.sdk.core import Config
import pandas as pd
import numpy as np
import requests
import json
import base64

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