# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: an end-to-end example (Unity Catalog)
# MAGIC
# MAGIC This tutorial shows you how to train and register Models in Unity Catalog. Databricks includes a hosted MLflow Model Registry in Unity Catalog, compatible with the open-source MLflow Python client. Benefits include centralized access control, auditing, lineage, and model discovery across workspaces. For details about managing the model lifecycle with Unity Catalog, see ([AWS](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.gcp.databricks.com/en/machine-learning/manage-model-lifecycle/index.html)).
# MAGIC
# MAGIC This tutorial covers the following steps:
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train multiple models
# MAGIC - Explore hyperparameter sweep results with MLflow
# MAGIC - Register the best performing model in MLflow
# MAGIC - Apply the registered model to another dataset using a Spark UDF
# MAGIC
# MAGIC In this example, you build a model to predict the quality of Portuguese "Vinho Verde" wine based on the wine's physicochemical properties. 
# MAGIC
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC
# MAGIC ## Requirements
# MAGIC This notebook requires a cluster running Databricks Runtime 15.4 LTS ML or above.
# MAGIC
# MAGIC This notebook requires a workspace that has been enabled for Unity Catalog. A version for workspaces not enabled for Unity Catalog is available: ([AWS](https://docs.databricks.com/mlflow/end-to-end-example.html) | [Azure](https://docs.microsoft.com/azure/databricks/mlflow/end-to-end-example) | [GCP](https://docs.gcp.databricks.com/mlflow/end-to-end-example.html)).

# COMMAND ----------

# MAGIC %md ## Unity Catalog setup
# MAGIC Set the catalog and schema where the model will be registered. You must have USE CATALOG privilege on the catalog, and CREATE MODEL and USE SCHEMA privileges on the schema. Change the catalog and schema here if necessary.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "")
dbutils.widgets.text("schema_name", "")

# COMMAND ----------

# Training Wine Quality Model with SHAP Explanations

catalog_name = dbutils.widgets.get("catalog_name")
schema_name = dbutils.widgets.get("schema_name")

# Create schema if it doesn't exist
query = f"""
CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
"""
spark.sql(query)

# COMMAND ----------

# Set up MLflow with Unity Catalog
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap
import time
import json
from mlflow.deployments import get_deploy_client
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")
CATALOG_NAME = catalog_name
SCHEMA_NAME = schema_name

# COMMAND ----------

conda_env = {
    'name': 'wine-quality-env',
    'channels': ['conda-forge', 'defaults'],
    'dependencies': [
        'python=3.11.11',
        'pip<=23.3.1',
        {
            'pip': [
                'mlflow==2.13.1',
                'shap==0.44.0',
                'xgboost==2.0.3',
                'scikit-learn==1.3.0',
                'pandas==1.5.3',
                'numpy==1.23.5',
                'cloudpickle==2.2.1',
                'pyarrow==14.0.1',
                'requests==2.31.0'
            ]
        }
    ]
}

# COMMAND ----------

# Download and prepare the wine-quality dataset
white_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

# Add binary color feature
white_wine['is_red'] = 0
red_wine['is_red'] = 1

# Combine datasets
wine = pd.concat([white_wine.rename(columns=lambda x: x.replace(' ', '_')), red_wine.rename(columns=lambda x: x.replace(' ', '_'))], ignore_index=True)

# Define features and target
X = wine.drop("quality", axis=1)
y = wine["quality"]

# Create binary classification target (good quality = 1, 0 otherwise)
y_binary = (y >= 7).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Save the data for app usage
wine_data = wine.drop("quality", axis=1)
table_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_data"
spark_df = spark.createDataFrame(wine_data)
(spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", True).saveAsTable(table_name))

# COMMAND ----------

input_example = pd.DataFrame([{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11,
    "total_sulfur_dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
    "is_red": 1
}])

# COMMAND ----------

# Define a custom model wrapper that includes SHAP explanations
import mlflow.pyfunc

# Define a custom model wrapper that includes SHAP explanations
class SHAPModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.feature_names = None
        
    def load_context(self, context):
        """Load context when model is loaded with dependency management"""
        try:
            # Load background data with pandas
            background_data = pd.read_csv(context.artifacts["background_data"])
            self.feature_names = background_data.columns.tolist()
            
            # Initialize explainer with numpy-backed data
            background_np = background_data.values.astype(np.float32)
            self.explainer = shap.TreeExplainer(self.model, background_np)
        except Exception as e:
            print(f"Error initializing SHAP explainer: {str(e)}")
            self.explainer = shap.TreeExplainer(self.model)
    
    def predict(self, context, model_input):
        """Generate predictions with SHAP explanations"""
        # Convert input to pandas DataFrame with proper types
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input).astype(np.float32)
            
        # Ensure feature names are stored
        if self.feature_names is None:
            self.feature_names = model_input.columns.tolist()
            
        # Make predictions
        predictions = self.model.predict_proba(model_input)[:, 1]  # Get probabilities for class 1
        
        # Get SHAP values using numpy array for efficiency
        shap_values = self.explainer.shap_values(model_input.values.astype(np.float32))
        
        # Convert outputs to Python native types for JSON serialization
        return {
            'predictions': predictions.tolist(),
            'shap_values': np.array(shap_values).tolist(),
            'feature_names': self.feature_names
        }

# COMMAND ----------

# Train the model with early stopping
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='auc'
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

# Create a background dataset for SHAP (keep small for serving)
background_data = X_train.sample(min(50, len(X_train)), random_state=42)
background_data.to_csv("/tmp/background_data.csv", index=False)

# Create the SHAP-enabled model wrapper
shap_model = SHAPModel(model)

# Infer model signature with proper input/output schema
signature = infer_signature(
    X_train,
    pd.DataFrame({
        'predictions': np.zeros(len(X_train)),
        'shap_values': [np.zeros(X_train.shape[1]).tolist() for _ in range(len(X_train))],
        'feature_names': [X_train.columns.tolist()] * len(X_train)
    })
)

# COMMAND ----------

# Log model with explicit environment and artifacts
with mlflow.start_run(run_name="wine_quality_with_shap") as run:
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Log model with all dependencies
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=shap_model,
        artifacts={"background_data": "/tmp/background_data.csv"},
        conda_env=conda_env,
        signature=signature,
        registered_model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_quality_with_shap",
        input_example=input_example
    )
    
    # Calculate metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    mlflow.log_metric("auc", auc)
    
    # Log training artifacts
    mlflow.log_text(json.dumps(dict(model.get_params())), "model_params.json")

# Register model in Unity Catalog
model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_quality_with_shap"
model_version = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name).version

# Set model alias
client = mlflow.MlflowClient()
client.set_registered_model_alias(model_name, "Champion", model_version)

print(f"Model version {model_version} registered with AUC: {auc:.4f}")

# COMMAND ----------

# Validate deployment environment locally
local_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
try:
    test_sample = X_test.iloc[:1]
    local_pred = local_model.predict(test_sample)
    print("Local test prediction:", local_pred)
except Exception as e:
    print(f"Local validation failed: {str(e)}")
    raise

# COMMAND ----------

# ✅ Deploy with small CPU + scale-to-zero enabled
deploy_client = get_deploy_client("databricks")

endpoint_name = f"{CATALOG_NAME}-{SCHEMA_NAME}-wine-endpoint"
endpoint_config = {
    "served_entities": [{
        "entity_name": model_name,
        "entity_version": model_version,
        "workload_size": "Small",  # Use Small CPU
        "scale_to_zero_enabled": True,  # Enable scale-to-zero
        "environment_vars": {
            "MLFLOW_DISABLE_ENV_CREATION": "false",
            "DISABLE_NGINX": "true",
            "GRPC_ENABLE_VERSION_CHECK": "false"
        }
    }],
    "inference_table": {
        "enabled": True,
        "destination": f"{CATALOG_NAME}.{SCHEMA_NAME}.wine_inference_logs"
    }
}

try:
    deploy_client.create_endpoint(
        name=endpoint_name,
        config=endpoint_config
    )
except Exception as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        deploy_client.update_endpoint(
            endpoint=endpoint_name,
            config=endpoint_config
        )
    else:
        raise

print("✅ Deployment initiated with small CPU and scale-to-zero enabled.")
