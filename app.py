from dash import Dash, html, dcc, callback, Output, Input, State, ALL, dash_table, ctx
import dash_bootstrap_components as dbc
from utils import read_table_data, get_column_stats
import pandas as pd
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Dash app
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
    
    # Create a data table component
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in df.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '8px'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
except Exception as e:
    logger.error(f"Error in layout creation: {str(e)}", exc_info=True)
    controls = [html.Div(f"Error loading data: {str(e)}")]
    table = html.Div()

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

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('What-If Simulation App', className="text-center my-4"),
            html.H2('Input Controls', className="text-center mb-4"),
            *controls,
            html.H2('Sample Data', className="text-center my-4"),
            table
        ])
    ])
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True) 