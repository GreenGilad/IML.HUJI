# Basic imports and settings for working with data
import numpy as np
import pandas as pd

# Imports and settings for plotting of graphs
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

pio.templates["custom"] = go.layout.Template(
    layout=go.Layout(
        margin=dict(l=20, r=20, t=40, b=0)
    )
)
pio.templates.default = "simple_white+custom"




def dash_show(fig):
    from jupyter_dash import JupyterDash
    import dash_core_components as dcc
    import dash_html_components as html
    app = JupyterDash(__name__)
    app.layout = html.Div([
        dcc.Graph(id='graph', figure=fig)
    ], style={'display': 'inline-block', 'width': '100%', "height": "100%"})
    app.run_server(mode='inline')
    

def save_animated_gif(frames, filename, frame_duration=100):
    import gif

    @gif.frame
    def plot(fr):
        fig = go.Figure(data=fr["data"], layout=fr["layout"])
        return fig
    
    frames = [plot(fr) for fr in frames]
    gif.save(frames, filename, duration=frame_duration)