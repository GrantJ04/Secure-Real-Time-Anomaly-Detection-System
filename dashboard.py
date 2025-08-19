import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import visualizeResults as viz

def db(recErrors, threshold, allLabels, predAll, scores):
    # Use a Bootstrap theme for a modern look
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    app.title = "Secure Anomaly Detection Dashboard"

    # Layout
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col(html.H1("Secure Anomaly Detection Dashboard",
                            style={'textAlign': 'center', 'marginBottom': '20px'}))
        ]),

        # First row of graphs
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=viz.plotRecErrorDist(recErrors, threshold)),
                             body=True, style={'marginBottom': '20px'}), width=6),
            dbc.Col(dbc.Card(dcc.Graph(figure=viz.plotErrorOverTime(recErrors, allLabels, predAll, threshold)),
                             body=True, style={'marginBottom': '20px'}), width=6),
        ]),

        # Second row of graphs
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=viz.plotConfusionMatrix(allLabels, predAll)),
                             body=True, style={'marginBottom': '20px'}), width=6),
            dbc.Col(dbc.Card(dcc.Graph(figure=viz.plotPrecisionRecallCurve(allLabels, scores)),
                             body=True, style={'marginBottom': '20px'}), width=6),
        ]),

    ], fluid=True, style={'padding': '20px', 'backgroundColor': '#1e1e1e', 'fontFamily': 'Arial'})

    print("Dash app running on http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)
