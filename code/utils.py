def dash_show(fig):
    from jupyter_dash import JupyterDash
    import dash_core_components as dcc
    import dash_html_components as html
    app = JupyterDash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='graph', figure=fig)
    ], style={'display': 'inline-block', 'width': '100%', "height": "100%"})
    app.run_server(mode='inline')