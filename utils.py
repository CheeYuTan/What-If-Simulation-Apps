import os
from databricks import sql
from databricks.sdk.core import Config
import pandas as pd
import numpy as np
from config import SAMPLE_TABLE_PATH, DATABRICKS_HOST

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
        server_hostname=DATABRICKS_HOST,  # Use hostname from config
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
        # Split the table path into catalog, schema, and table
        catalog, schema, table = SAMPLE_TABLE_PATH.split('.')
        
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
        # Split the table path into catalog, schema, and table
        catalog, schema, table = SAMPLE_TABLE_PATH.split('.')
        
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