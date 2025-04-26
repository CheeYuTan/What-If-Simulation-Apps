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
    create_prediction_display
)
import pandas as pd
import logging
import numpy as np
import io
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app with store
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    
except Exception as e:
    logger.error(f"Error in layout creation: {str(e)}", exc_info=True)
    controls = [html.Div(f"Error loading data: {str(e)}")]

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

# Define the app layout
app.layout = dbc.Container([
    dcc.Store(id='prediction-store'),  # Store for prediction results
    dbc.Row([
        dbc.Col([
            html.H1('What-If Simulation App', className="text-center my-4"),
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 