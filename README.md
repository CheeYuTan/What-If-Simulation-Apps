# Wine Quality Predictor (Dash + Databricks)

## Overview
This application is a web-based dashboard for predicting wine quality using machine learning models hosted on Databricks. It supports both real-time single predictions and batch inference jobs, with user-specific file upload and download functionality leveraging Unity Catalog Volumes. The app also features an integrated AI agent that explains predictions, provides suggestions, and helps users interpret results.

## Features
- **Real-Time Prediction:** Interactive controls for single wine quality prediction with SHAP explanations.
- **AI Agent Explanations:** An integrated agent interprets predictions, explains feature impacts, and offers actionable suggestions based on the results.
- **Batch Inference:** Upload CSV files for batch predictions, validate files, and run Databricks jobs.
- **User-Specific File Management:** Uploaded and downloaded files are organized in user-specific folders (by email) in Databricks Volumes.
- **Download Results:** Users can view and download their batch inference results from a dedicated subpage.
- **Job Status Tracking:** Monitor the status of batch inference jobs.

## Environment Variables

### Set Automatically by the App (from deployment config)
These variables are typically set by the app or deployment environment and do not need to be specified manually:

- `DATABRICKS_HOST`: Databricks workspace hostname
- `DATABRICKS_WAREHOUSE_ID`: SQL Warehouse ID
- `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET`: OAuth credentials
- `DATABRICKS_BATCH_INFERENCE_JOB_ID`: Databricks Job ID for batch inference
- `DATABRICKS_SERVING_ENDPOINT`: Model serving endpoint name
- `INTERPRETER_SERVING_ENDPOINT`: LLM interpreter endpoint name

### User-Specified (must be set manually)
These variables must be specified by the user (e.g., in a `.env` file or deployment config):

- `CATALOG`: Unity Catalog name
- `SCHEMA`: Unity Catalog schema
- `VOLUME_UPLOAD_PATH`: Upload volume name
- `VOLUME_DOWNLOAD_PATH`: Download volume name
- `SAMPLE_TABLE_PATH`: Table path for sample data

## Usage
- **Single Prediction:**
  - Use the "Real Time Prediction" page to input wine features and get a prediction with explanations.
  - The AI agent will interpret the prediction, explain feature contributions, and provide suggestions.
- **Batch Inference:**
  - Go to "Batch Inference" to upload a CSV file (with required columns), validate, and run a batch job.
  - Each user's files are stored in a folder named after their email.
  - After the job completes, results are saved in the user's download folder.
- **Download Results:**
  - Visit "Download Batch Inference File" to see and download your batch results. Only your files are shown.

## Batch Inference Download
- The app lists all files in `/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_DOWNLOAD_PATH}/{user_email}`.
- Click the ⬇️ Download link to download your result file as a CSV.

## Troubleshooting
- **File Not Found:** Ensure your batch job has completed and results are saved in your user folder.
- **Job Fails:** Check Databricks job logs for errors (e.g., missing folders, invalid file format).
- **Environment Variables:** Missing or incorrect variables will cause errors—double-check your configuration.
- **Permissions:** Ensure your Databricks user has access to the relevant volumes and jobs. 