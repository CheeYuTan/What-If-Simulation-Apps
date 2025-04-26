import os

# List of environment variables to check
env_vars = [
    "DATABRICKS_WAREHOUSE_ID",
    "DATABRICKS_SERVING_ENDPOINT",
    "SAMPLE_TABLE_PATH",
    "MLFLOW_TRACKING_URI",
    "INTERPRETER_SERVING_ENDPOINT",
    "CATALOG",
    "SCHEMA",
    "VOLUME_UPLOAD_PATH",
    "VOLUME_DOWNLOAD_PATH",
    "DATABRICKS_BATCH_INFERENCE_JOB"
]

print("Environment Variables:")
print("=" * 50)
for var in env_vars:
    value = os.getenv(var)
    if value:
        print(f"{var}: {value}")
    else:
        print(f"{var}: Not set") 