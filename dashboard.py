import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import visualizeResults as viz
import pandas as pd

def db(recErrors, threshold, allLabels, predAll, recErrors_rf, finance_pred_file="finance_preds.csv"):

    # Load financial predictions
    df_fin = pd.read_csv(finance_pred_file)
    y_fin = df_fin['y_true'].values
    y_pred_fin = df_fin['y_pred'].values

    app = dash.Dash(__name__)
    app.title = "Secure Anomaly Detection Dashboard"

    # Apply dark theme for Plotly and basic CSS
    app.layout = html.Div([
        html.H1("Secure Anomaly Detection Dashboard", style={
            'textAlign': 'center', 'color': 'white', 'marginBottom': '30px'}),
        dcc.Tabs(id="tabs", value='tab-rf', children=[
            dcc.Tab(label='RF Anomaly Detection', value='tab-rf', style={'color': 'black'}, selected_style={'color': 'white'}),
            dcc.Tab(label='Financial Fraud Detection', value='tab-finance', style={'color': 'black'}, selected_style={'color': 'white'})
        ]),
        html.Div(id='page-content', style={'padding': '20px'})
    ], style={'backgroundColor': '#111111', 'font-family': 'Arial, sans-serif'})

    @app.callback(
        Output('page-content', 'children'),
        [Input('tabs', 'value')]
    )
    def render_tab(tab):
        if tab == 'tab-rf':
            return html.Div([
                dcc.Graph(figure=viz.plotRecErrorDist(recErrors, threshold).update_layout(template='plotly_dark')),
                dcc.Graph(figure=viz.plotErrorOverTime(recErrors, allLabels, predAll, threshold).update_layout(template='plotly_dark')),
                dcc.Graph(figure=viz.plotConfusionMatrix(allLabels, predAll).update_layout(template='plotly_dark')),
                dcc.Graph(figure=viz.plotPrecisionRecallCurve(allLabels, recErrors_rf).update_layout(template='plotly_dark'))
            ])
        elif tab == 'tab-finance':
            return html.Div([
                dcc.Graph(figure=viz.plotFinancialCounts(y_fin, y_pred_fin).update_layout(
                    template='plotly_dark',
                    title='Fraud Counts per Bucket',
                    xaxis_title='Data Bucket',
                    yaxis_title='Number of Fraud Cases',
                    font=dict(color='white')
                ))
            ])

    print("Dash app running on http://127.0.0.1:8050")
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)
