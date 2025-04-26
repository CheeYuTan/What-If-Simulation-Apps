from dash import Dash, html, dcc, callback, Output, Input, State, ALL, dash_table, ctx, no_update
import dash_bootstrap_components as dbc
from utils import (
    read_table_data,
    get_column_stats,
    send_endpoint_request,
    create_display_content,
    format_llm_prompt,
    send_interpreter_request,
    stream_prediction_results,
    create_prediction_display,
    upload_file_to_volume_uc_files_api,
    validate_uploaded_file,
    run_batch_inference_job,
    check_job_status,
    sqlQuery
)
import pandas as pd
import logging
import numpy as np
import io
from datetime import datetime
import base64
import os
from flask import request
from databricks.sdk.core import Config
from databricks import sql as dbsql
import dash  # Add this at the top with other imports

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app with store
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Real Time Prediction", href="/")),
        dbc.NavItem(dbc.NavLink("Batch Inference", href="/batch-inference")),
        dbc.NavItem(dbc.NavLink("Download Batch Inference File", href="/batch-inference/download")),
    ],
    brand="Wine Quality Predictor",
    brand_href="/",
    color="primary",
    dark=True,
)

# Define the single prediction page layout
def single_prediction_layout():
    # Read the data and get column stats
    try:
        logger.info("Reading table data...")
        df = read_table_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        logger.info("Getting column stats...")
        column_stats = get_column_stats()
        logger.info(f"Column stats: {column_stats}")
        
        # Create controls for each column
        controls = []
        current_row = []
        
        for column, stats in column_stats.items():
            logger.info(f"Creating control for column: {column}")
            
            if column == 'is_red':
                # Special handling for is_red as categorical
                control = dbc.Card([
                    dbc.CardHeader(column),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id=f'dropdown-{column}',
                            options=[
                                {'label': 'White Wine', 'value': 0},
                                {'label': 'Red Wine', 'value': 1}
                            ],
                            value=0
                        )
                    ])
                ], className="mb-3 h-100")  # Added h-100 for consistent card heights
            elif stats['type'] == 'numeric':
                # Create slider for numerical columns
                min_val = float(stats['min'])
                max_val = float(stats['max'])
                
                # Calculate appropriate step size
                range_val = max_val - min_val
                if range_val <= 1:
                    step_val = 0.001
                elif range_val <= 10:
                    step_val = 0.01
                elif range_val <= 100:
                    step_val = 0.1
                else:
                    step_val = 1.0
                
                # Determine number of decimal places based on step value
                if step_val < 0.01:
                    decimal_places = 3
                elif step_val < 0.1:
                    decimal_places = 2
                elif step_val < 1:
                    decimal_places = 1
                else:
                    decimal_places = 0
                    
                # Format values with consistent decimal places
                min_str = f"{min_val:.{decimal_places}f}"
                max_str = f"{max_val:.{decimal_places}f}"
                current_val = float(stats['current_min'])
                current_str = f"{current_val:.{decimal_places}f}"
                
                control = dbc.Card([
                    dbc.CardHeader(column),
                    dbc.CardBody([
                        dcc.Slider(
                            id={'type': 'slider', 'index': column},
                            min=float(min_str),
                            max=float(max_str),
                            value=float(current_str),
                            step=step_val,
                            marks={
                                float(min_str): min_str,
                                float(max_str): max_str
                            },
                            included=True,
                            updatemode='drag'
                        ),
                        dbc.Input(
                            id={'type': 'input', 'index': column},
                            type="number",
                            min=float(min_str),
                            max=float(max_str),
                            step=step_val,
                            value=float(current_str),
                            className="mt-2"
                        )
                    ])
                ], className="mb-3 h-100")  # Added h-100 for consistent card heights
            else:
                # Create dropdown for other categorical columns
                control = dbc.Card([
                    dbc.CardHeader(column),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id=f'dropdown-{column}',
                            options=[{'label': str(val), 'value': val} for val in stats['values']],
                            value=stats['values'][0]
                        )
                    ])
                ], className="mb-3 h-100")  # Added h-100 for consistent card heights
            
            current_row.append(control)
            
            # If we have 3 controls in the current row, add them to controls and start a new row
            if len(current_row) == 3:
                controls.append(dbc.Row([
                    dbc.Col(current_row[0], width=4),
                    dbc.Col(current_row[1], width=4),
                    dbc.Col(current_row[2], width=4)
                ], className="mb-3"))
                current_row = []
        
        # Add any remaining controls in the last row
        if current_row:
            # Add empty columns if needed to maintain layout
            while len(current_row) < 3:
                current_row.append(None)
            
            controls.append(dbc.Row([
                dbc.Col(current_row[0] if current_row[0] else "", width=4),
                dbc.Col(current_row[1] if current_row[1] else "", width=4),
                dbc.Col(current_row[2] if current_row[2] else "", width=4)
            ], className="mb-3"))
        
        return dbc.Container([
            dcc.Store(id='prediction-store'),  # Store for prediction results
            dbc.Row([
                dbc.Col([
                    html.H1('Real Time Prediction', className="text-center my-4"),
                    html.H2('Input Controls', className="text-center mb-4"),
                    *controls,
                    dbc.Row([
                        dbc.Col([
                            dbc.Button(
                                "Make Prediction",
                                id="predict-button",
                                color="primary",
                                size="lg",
                                className="w-100 mb-2"
                            ),
                            dbc.Button(
                                "Explain Results",
                                id="explain-button",
                                color="info",
                                size="lg",
                                className="w-100 mb-2",
                                disabled=True
                            ),
                            dbc.Button(
                                "Download Results",
                                id="download-button",
                                color="success",
                                size="lg",
                                className="w-100 mb-2",
                                disabled=True
                            ),
                            dcc.Download(id="download-results"),
                            html.Div(id="prediction-output"),
                            html.Div(id="explanation-output")
                        ], width={"size": 6, "offset": 3})
                    ], className="mt-4")
                ])
            ])
        ], fluid=True)
    except Exception as e:
        logger.error(f"Error in layout creation: {str(e)}", exc_info=True)
        return html.Div(f"Error loading data: {str(e)}")

# Define the batch inference page layout
def batch_inference_layout():
    return html.Div([
        html.H1("Batch Inference", className="text-center my-4"),
        html.Div([
            dbc.Card([
                dbc.CardHeader("Upload CSV File"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='output-data-upload'),
                    html.Div(id='validate-button-container', style={'display': 'none'}, children=[
                        dbc.Button(
                            "Validate File",
                            id="validate-button",
                            color="primary",
                            size="lg",
                            className="w-100 mt-3"
                        )
                    ]),
                    html.Div(id='validation-output'),
                    html.Div(id='run-job-button-container', style={'display': 'none'}, children=[
                        dbc.Button(
                            "Run Batch Job",
                            id="run-job-button",
                            color="success",
                            size="lg",
                            className="w-100 mt-3"
                        )
                    ]),
                    html.Div(id='job-status-output'),
                    html.Div(id='check-status-button-container', style={'display': 'none'}, children=[
                        dbc.Button(
                            "Check Job Status",
                            id="check-status-button",
                            color="info",
                            size="lg",
                            className="w-100 mt-3"
                        )
                    ]),
                    html.Div(id='job-status-details')
                ])
            ])
        ])
    ])

# Define the download batch inference file layout
def download_batch_inference_file_layout():
    return html.Div([
        html.H1("Download Batch Inference File", className="text-center my-4"),
        html.P("Below are your available batch inference result files."),
        dbc.Button("Refresh File List", id="refresh-download-list", color="secondary", className="mb-3"),
        html.Div(id="download-file-list-output"),
        dcc.Download(id="download-batch-file"),
        html.Div(id="download-message", style={"display": "none"})
    ])

# Define the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/batch-inference/download':
        return download_batch_inference_file_layout()
    elif pathname == '/batch-inference':
        return batch_inference_layout()
    else:
        return single_prediction_layout()

# Define callbacks for syncing slider and input
@app.callback(
    Output({'type': 'slider', 'index': ALL}, 'value'),
    Output({'type': 'input', 'index': ALL}, 'value'),
    Input({'type': 'slider', 'index': ALL}, 'value'),
    Input({'type': 'input', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def sync_controls(slider_values, input_values):
    triggered = ctx.triggered_id
    
    if triggered['type'] == 'slider':
        return slider_values, slider_values
    else:
        return input_values, input_values

# Modify the prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Output('prediction-store', 'data'),
    Output('explain-button', 'disabled'),
    Output('download-button', 'disabled'),  # Add download button to outputs
    Input('predict-button', 'n_clicks'),
    State({'type': 'input', 'index': ALL}, 'value'),
    State({'type': 'input', 'index': ALL}, 'id'),
    State('dropdown-is_red', 'value'),
    prevent_initial_call=True
)
def make_prediction(n_clicks, input_values, input_ids, is_red_value):
    if n_clicks is None:
        return "", None, True, True
        
    try:
        # Create input data dictionary
        input_data = {}
        
        # Add numeric values from inputs
        for value, id_dict in zip(input_values, input_ids):
            column = id_dict['index']
            if column != 'is_red':  # Skip is_red as we get it from dropdown
                input_data[column] = float(value)
        
        # Add is_red value
        input_data['is_red'] = int(is_red_value)
        
        # Log the input data
        logger.info(f"Sending prediction request with input data: {input_data}")
        
        # Send request to endpoint
        result = send_endpoint_request(input_data)
        
        # Log the result
        logger.info(f"Received response from endpoint: {result}")
        
        # Handle nested response structure
        if isinstance(result, dict) and 'predictions' in result:
            predictions_data = result['predictions']
            if isinstance(predictions_data, dict) and 'predictions' in predictions_data:
                prediction = predictions_data['predictions'][0]
                shap_values = predictions_data.get('shap_values', [[]])[0]
                feature_names = predictions_data.get('feature_names', [])
                
                # Store results for explanation
                store_data = {
                    'input_data': input_data,
                    'prediction': prediction,
                    'shap_values': shap_values,
                    'feature_names': feature_names,
                    'interpretation': None  # Initialize interpretation as None
                }
                
                # Create prediction display
                alert_content = [
                    html.H4("Prediction Result", className="alert-heading"),
                    html.Hr(),
                    html.P(f"Quality Score: {float(prediction):.2f}", className="mb-0"),
                    html.Hr(),
                    html.H5("Feature Contributions (SHAP Values):", className="mt-3")
                ]
                
                # Add SHAP values if available
                if shap_values and feature_names:
                    # Sort features by absolute SHAP value
                    feature_impacts = list(zip(feature_names, shap_values))
                    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    for feature, value in feature_impacts:
                        alert_content.append(
                            html.P(f"{feature}: {value:.3f}",
                                  className="mb-1",
                                  style={'color': 'red' if value < 0 else 'green'})
                        )
                
                return (
                    dbc.Alert(alert_content, color="success", className="mt-3"),
                    store_data,
                    False,  # Enable explain button
                    False   # Enable download button
                )
            else:
                return (
                    dbc.Alert("Unexpected response format from the model", color="warning", className="mt-3"),
                    None,
                    True,
                    True
                )
        else:
            return (
                dbc.Alert("No predictions found in the model response", color="warning", className="mt-3"),
                None,
                True,
                True
            )
            
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        return (
            dbc.Alert(
                [
                    html.H4("Error", className="alert-heading"),
                    html.Hr(),
                    html.P(f"Error making prediction: {str(e)}", className="mb-0")
                ],
                color="danger",
                className="mt-3"
            ),
            None,
            True,
            True
        )

@app.callback(
    Output('explanation-output', 'children'),
    Output('prediction-store', 'data', allow_duplicate=True),  # Add store update
    Input('explain-button', 'n_clicks'),
    State('prediction-store', 'data'),
    prevent_initial_call=True
)
def get_explanation(n_clicks, store_data):
    if n_clicks is None or store_data is None:
        return no_update, no_update
        
    try:
        # Format the prompt
        prompt = format_llm_prompt(
            prediction=store_data['prediction'],
            input_features=store_data['input_data'],
            shap_values=store_data['shap_values'],
            feature_names=store_data['feature_names']
        )
        
        # Get explanation
        explanation = send_interpreter_request(prompt)
        
        # Update store data with interpretation
        store_data['interpretation'] = explanation
            
        # Create display
        return dbc.Alert(
            [
                html.H4("AI Interpretation", className="alert-heading"),
                html.Hr(),
                html.Div(
                    [html.P(line) for line in explanation.split('\n') if line.strip()],
                    style={
                        'white-space': 'pre-wrap',
                        'padding': '1rem',
                        'background-color': '#ffffff',
                        'border-radius': '0.25rem',
                        'border': '1px solid #dee2e6'
                    }
                )
            ],
            color="info",
            className="mt-3"
        ), store_data
            
    except Exception as e:
        logger.error(f"Error getting explanation: {str(e)}", exc_info=True)
        return dbc.Alert(
            f"Error getting explanation: {str(e)}",
            color="danger",
            className="mt-3"
        ), no_update

# Add download callback
@app.callback(
    Output("download-results", "data"),
    Input("download-button", "n_clicks"),
    State("prediction-store", "data"),
    prevent_initial_call=True
)
def download_results(n_clicks, stored_data):
    if not stored_data:
        return no_update
        
    try:
        # Create a DataFrame with input features
        input_df = pd.DataFrame([stored_data['input_data']])
        
        # Add prediction column
        input_df['prediction'] = stored_data['prediction']
        
        # Add SHAP values if available
        if stored_data['shap_values'] and stored_data['feature_names']:
            for feature, shap_value in zip(stored_data['feature_names'], stored_data['shap_values']):
                input_df[f'shap_{feature}'] = shap_value
        
        # Add AI interpretation if available
        if 'interpretation' in stored_data and stored_data['interpretation']:
            # Clean up the interpretation text (remove markdown and extra whitespace)
            interpretation_text = stored_data['interpretation']
            interpretation_text = interpretation_text.replace('**', '')  # Remove markdown bold
            interpretation_text = '\n'.join(line.strip() for line in interpretation_text.split('\n') if line.strip())
            input_df['ai_interpretation'] = interpretation_text
                
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert DataFrame to CSV
        csv_string = input_df.to_csv(index=False)
        
        return dict(
            content=csv_string,
            filename=f'prediction_results_{timestamp}.csv',
            type='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error creating download file: {str(e)}", exc_info=True)
        return no_update

# Add callback for file upload
@app.callback(
    Output('output-data-upload', 'children'),
    Output('upload-data', 'data'),  # Store the decoded content
    Output('validate-button-container', 'style'),  # Show/hide validate button
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    if contents is None:
        return no_update, no_update, {'display': 'none'}
    try:
        # Extract user email from headers
        email = request.headers.get('X-Forwarded-Email')
        if not email:
            raise ValueError("User email not found in request headers.")
        # Use only the date portion for the file prefix
        date_str = datetime.now().strftime("%Y%m%d")
        unique_filename = f"{date_str}_{filename}"
        folder_name = email
        # Get volume path from environment variables
        catalog = os.getenv('CATALOG')
        schema = os.getenv('SCHEMA')
        volume_name = os.getenv('VOLUME_UPLOAD_PATH')
        if not all([catalog, schema, volume_name]):
            raise ValueError("Missing required environment variables for volume path")
        # Construct the full volume path with user folder
        volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}/{folder_name}"
        saved_path, sample_content = upload_file_to_volume_uc_files_api(contents, volume_path, unique_filename, flask_request=request)
        # Store folder_name in the sample_content for later use (or use dcc.Store if needed)
        return dbc.Alert(
            [
                html.H4("File Upload Successful", className="alert-heading"),
                html.Hr(),
                html.P(f"File saved to: {saved_path}"),
                html.P(f"Filename: {filename}"),
                html.P(f"User Folder: {folder_name}")
            ],
            color="success",
            className="mt-3"
        ), {'sample_content': sample_content, 'folder_name': folder_name, 'unique_filename': unique_filename}, {'display': 'block'}
    except Exception as e:
        return dbc.Alert(
            [
                html.H4("Error Uploading File", className="alert-heading"),
                html.Hr(),
                html.P(f"Error: {str(e)}")
            ],
            color="danger",
            className="mt-3"
        ), no_update, {'display': 'none'}

# Add callback for file validation
@app.callback(
    Output('validation-output', 'children'),
    Output('validate-button', 'disabled'),
    Output('run-job-button-container', 'style'),  # Show/hide run job button
    Input('validate-button', 'n_clicks'),
    State('upload-data', 'data'),  # Get the cached content (now a dict)
    prevent_initial_call=True
)
def validate_file(n_clicks, cached_content):
    if n_clicks is None or not cached_content:
        return no_update, no_update, {'display': 'none'}
    try:
        # Extract the sample_content string from the dict
        if isinstance(cached_content, dict) and 'sample_content' in cached_content:
            content_to_validate = cached_content['sample_content']
        else:
            content_to_validate = cached_content
        # Validate the file using the extracted content
        is_valid, validation_message = validate_uploaded_file(content_to_validate)
        if is_valid:
            return dbc.Alert(
                [
                    html.H4("Validation Successful", className="alert-heading"),
                    html.Hr(),
                    html.P(validation_message)
                ],
                color="success",
                className="mt-3"
            ), True, {'display': 'block'}  # Show run job button on success
        else:
            return dbc.Alert(
                [
                    html.H4("Validation Failed", className="alert-heading"),
                    html.Hr(),
                    html.P(validation_message)
                ],
                color="danger",
                className="mt-3"
            ), False, {'display': 'none'}  # Hide run job button on failure
    except Exception as e:
        return dbc.Alert(
            [
                html.H4("Error During Validation", className="alert-heading"),
                html.Hr(),
                html.P(f"Error: {str(e)}")
            ],
            color="danger",
            className="mt-3"
        ), False, {'display': 'none'}  # Hide run job button on error

# Add callback for running batch job
@app.callback(
    Output('job-status-output', 'children'),
    Output('run-job-button', 'disabled'),
    Output('check-status-button-container', 'style'),  # Show/hide check status button
    Input('run-job-button', 'n_clicks'),
    State('upload-data', 'data'),  # Now using the dict with folder_name and unique_filename
    prevent_initial_call=True
)
def run_batch_job(n_clicks, upload_data):
    if n_clicks is None or not upload_data:
        return no_update, no_update, {'display': 'none'}
    try:
        # Extract filename and folder_name from upload_data
        unique_filename = upload_data.get('unique_filename')
        folder_name = upload_data.get('folder_name')
        if not unique_filename or not folder_name:
            return dbc.Alert(
                "Could not find the uploaded file path or user folder",
                color="danger",
                className="mt-3"
            ), False, {'display': 'none'}
        # Run the batch job with both filename and folder_name
        run_id = run_batch_inference_job(unique_filename, folder_name)
        return dbc.Alert(
            [
                html.H4("Batch Job Started", className="alert-heading"),
                html.Hr(),
                html.P(f"Job Run ID: {run_id}"),
                html.P("The job is running in the background. Click 'Check Job Status' to monitor progress.")
            ],
            color="success",
            className="mt-3"
        ), True, {'display': 'block'}  # Show check status button
    except Exception as e:
        return dbc.Alert(
            [
                html.H4("Error Starting Batch Job", className="alert-heading"),
                html.Hr(),
                html.P(f"Error: {str(e)}")
            ],
            color="danger",
            className="mt-3"
        ), False, {'display': 'none'}

# Add callback for checking job status
@app.callback(
    Output('job-status-details', 'children'),
    Input('check-status-button', 'n_clicks'),
    State('job-status-output', 'children'),
    prevent_initial_call=True
)
def get_job_status(n_clicks, job_status_output):
    if n_clicks is None or not job_status_output:
        return no_update
    try:
        # Extract run ID from the job status output
        run_id = None
        for child in job_status_output['props']['children']:
            if isinstance(child, dict) and 'props' in child and 'children' in child['props']:
                if isinstance(child['props']['children'], str) and child['props']['children'].startswith('Job Run ID:'):
                    run_id = child['props']['children'].split('Job Run ID: ')[1].strip()
                    break
        if not run_id:
            return dbc.Alert(
                "Could not find the job run ID",
                color="danger",
                className="mt-3"
            )
        # Check job status
        status_info = check_job_status(run_id)
        details = []
        details.append(html.H4("Job Status", className="alert-heading"))
        details.append(html.Hr())
        # Start Time
        if status_info.get('start_time'):
            details.append(html.P(f"Start Time: {status_info['start_time']}", className="mb-1"))
        # Life Cycle State and State Message
        state = status_info.get('state')
        state_message = status_info.get('state_message')
        details.append(html.P(f"Life Cycle State: {state}", className="mb-1"))
        if state_message:
            details.append(html.P(f"State Message: {state_message}", className="mb-1"))
        # status.state (from tasks or status)
        status_state = None
        run_page_url = None
        if 'tasks' in status_info and status_info['tasks']:
            task = status_info['tasks'][0]
            if 'status' in task and 'state' in task['status']:
                status_state = task['status']['state']
            if 'run_page_url' in task:
                run_page_url = task['run_page_url']
        if status_state:
            details.append(html.P(f"Task Status: {status_state}", className="mb-1"))
        # Optionally, show run page URL if available
        if run_page_url:
            details.append(html.A("View in Databricks UI", href=run_page_url, target="_blank"))
        color = 'info'
        return dbc.Alert(details, color=color, className="mt-3")
    except Exception as e:
        return dbc.Alert(
            [
                html.H4("Error Checking Job Status", className="alert-heading"),
                html.Hr(),
                html.P(f"Error: {str(e)}")
            ],
            color="danger",
            className="mt-3"
        )

# Add callback to list files in the user's download folder
@app.callback(
    Output("download-file-list-output", "children"),
    Input("refresh-download-list", "n_clicks"),
    prevent_initial_call=True
)
def list_user_download_files(n_clicks):
    from flask import request
    import os
    # Get user email from header
    email = request.headers.get('X-Forwarded-Email')
    if not email:
        return dbc.Alert("User email not found in request headers.", color="danger")
    # Get volume path from env
    catalog = os.getenv('CATALOG')
    schema = os.getenv('SCHEMA')
    volume_name = os.getenv('VOLUME_DOWNLOAD_PATH')
    if not all([catalog, schema, volume_name]):
        return dbc.Alert("Missing required environment variables for volume path.", color="danger")
    # Construct the user folder path
    user_folder_path = f"/Volumes/{catalog}/{schema}/{volume_name}/{email}"
    # List files in the user folder
    try:
        list_query = f"LIST '{user_folder_path}'"
        df = sqlQuery(list_query)
        if df.empty:
            return dbc.Alert("No files found in your download folder.", color="info")
        # Display as table with a styled download action
        df['Download'] = ["⬇️ Download" for _ in range(len(df))]
        return dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
            row_selectable=False,
            cell_selectable=True,
            id="batch-download-table",
            style_data_conditional=[
                {
                    'if': {'column_id': 'Download'},
                    'color': '#1976d2',
                    'textDecoration': 'underline',
                    'cursor': 'pointer',
                    'fontWeight': 'bold',
                }
            ],
        )
    except Exception as e:
        return dbc.Alert(f"Error listing files: {str(e)}", color="danger")

# Ensure the subpage is shown on first visit (trigger refresh)
@app.callback(
    Output("refresh-download-list", "n_clicks"),
    Input("page-content", "children"),
    prevent_initial_call=True
)
def trigger_refresh_on_subpage(children):
    # If the download subpage is loaded, trigger the refresh button
    if isinstance(children, dict) and 'props' in children and 'children' in children['props']:
        if any(
            isinstance(grandchild, dict) and 'props' in grandchild and 'Download Batch Inference File' in str(grandchild['props'].get('children', ''))
            for grandchild in children['props']['children']
        ):
            return 1
    return dash.no_update

# Add callback to handle file download when a cell in the Download column is clicked
@app.callback(
    Output("download-batch-file", "data"),
    Input("batch-download-table", "active_cell"),
    State("batch-download-table", "data"),
    prevent_initial_call=True
)
def download_selected_batch_file(active_cell, table_data):
    if not active_cell or not table_data:
        return dash.no_update
    col = active_cell.get('column_id')
    row = active_cell.get('row')
    if col != 'Download':
        return dash.no_update
    # Get the file path from the row data (usually the 'file' or 'name' column)
    row_data = table_data[row]
    # Try to find the file path column (commonly 'file', 'name', or 'file_path')
    file_path = None
    for key in row_data:
        if key.lower() in ['file', 'name', 'file_path', 'path']:
            file_path = row_data[key]
            break
    if not file_path:
        return dash.no_update
    # Read the file from the volume using sqlQuery GET
    # Download the file to /tmp
    local_temp_path = f"/tmp/download_{os.path.basename(file_path)}"
    query = f"GET '{file_path}' TO '{local_temp_path}'"
    try:
        cfg = Config()
        host = os.getenv('DATABRICKS_HOST')
        with dbsql.connect(
            server_hostname=host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate,
            staging_allowed_local_path="/tmp"
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
        # Read the file content
        with open(local_temp_path, 'r') as f:
            content = f.read()
        # Clean up
        os.remove(local_temp_path)
        # Return as downloadable CSV
        return dict(content=content, filename=os.path.basename(file_path), type='text/csv')
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 