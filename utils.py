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


class AnimationButtons():
    @staticmethod
    def play_scatter(frame_duration = 500, transition_duration = 300):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": False},
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "quadratic-in-out"}}])
    
    @staticmethod
    def play(frame_duration = 1000, transition_duration = 0):
        return dict(label="Play", method="animate", args=
                    [None, {"frame": {"duration": frame_duration, "redraw": True},
                            "mode":"immediate",
                            "fromcurrent": True, "transition": {"duration": transition_duration, "easing": "linear"}}])
    
    @staticmethod
    def pause():
        return dict(label="Pause", method="animate", args=
                    [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
    
    @staticmethod
    def slider(frame_names):       
        steps= [dict(args=[[i], dict(frame={'duration': 300, 'redraw': False}, mode="immediate", transition= {'duration': 300})],
                           label=i+1, method="animate")
                for i, n in enumerate(frame_names)]
        
        return [dict(yanchor="top", xanchor="left",
                     currentvalue={'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                     transition={'duration': 0, 'easing': 'linear'},
                     pad= {'b': 10, 't': 50},
                     len=0.9, x=0.1, y=0,
                     steps=steps)]


custom = [[0.0, "rgb(165,0,38)"],
          [0.1111111111111111, "rgb(215,48,39)"],
          [0.2222222222222222, "rgb(244,109,67)"],
          [0.3333333333333333, "rgb(253,174,97)"],
          [0.4444444444444444, "rgb(254,224,144)"],
          [0.5555555555555556, "rgb(224,243,248)"],
          [0.6666666666666666, "rgb(171,217,233)"],
          [0.7777777777777778, "rgb(116,173,209)"],
          [0.8888888888888888, "rgb(69,117,180)"],
          [1.0, "rgb(49,54,149)"]]

class_symbols = np.array(["circle", "x", "diamond"])
class_colors = lambda n: [custom[i] for i in np.linspace(0, len(custom)-1, n).astype(int)]

def decision_surface(predict, xrange, yrange, density=120, dotted=False, colorscale=custom, showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()])

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers", marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False), hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=showscale)


   
def animation_to_gif(fig, filename, frame_duration=100, width=1200, height=800):
    import gif
    @gif.frame
    def plot(f, i):
        f_ = go.Figure(data=f["frames"][i]["data"], layout=f["layout"])
        f_["layout"]["updatemenus"] = []
        f_.update_layout(title=f["frames"][i]["layout"]["title"], width=width, height=height)
        return f_

    gif.save([plot(fig, i) for i in range(len(fig["frames"]))], filename, duration=frame_duration)



def create_data_bagging_utils(d = 4, number_of_members = 1, n_samples = 1000):
    
    def sample_beta(limit1, limit2):
        margin1 = limit1 + (limit2 - limit1)*0.45
        margin2 = limit2 - (limit2 - limit1)*0.45
        beta = np.random.uniform(margin1, margin2)
        return beta

  # Creates n samples
    samples = np.random.uniform(size=(n_samples, 2))

    samples_of_half = "samples_of_half"
    x_1 = "x_1"; x_2 = "x_2"; y_1 = "y_1"; y_2 = "y_2"; tag = "tag"
    list_of_array = {0: {samples_of_half : samples, x_1 : 0, x_2 : 1, y_1 : 0, y_2 : 1}}

    for i in range(0, d):
        built_list =  {}
        for sample_curr_i, sample_curr in enumerate(list_of_array.values()):
      # Choose if we want to split x axis or y axis
            dim_half = np.random.choice([0,1])

            dots_coords = sample_curr[samples_of_half]
            if (dim_half == 0):
                beta = sample_beta(sample_curr[x_1], sample_curr[x_2])
                built_list[sample_curr_i*2] = {samples_of_half: dots_coords[dots_coords[:,0] <= beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : beta,
                                       y_1 : sample_curr[y_1],
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}

                built_list[sample_curr_i*2 + 1] = {samples_of_half: dots_coords[dots_coords[:,0] > beta],
                                       x_1 : beta,
                                       x_2 : sample_curr[x_2],
                                       y_1 : sample_curr[y_1],
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}
            else:
                beta = sample_beta(sample_curr[y_1], sample_curr[y_2])
                built_list[sample_curr_i*2] = {samples_of_half: dots_coords[dots_coords[:,1] <= beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : sample_curr[x_2],
                                       y_1 : sample_curr[y_1],
                                       y_2 : beta,
                                       tag : np.random.choice([0, 1]).astype(int)}

                built_list[sample_curr_i*2 + 1] = {samples_of_half: dots_coords[dots_coords[:,1] > beta],
                                       x_1 : sample_curr[x_1],
                                       x_2 : sample_curr[x_2],
                                       y_1 : beta,
                                       y_2 : sample_curr[y_2],
                                       tag : np.random.choice([0, 1]).astype(int)}


        list_of_array =  built_list
    samples = np.vstack([samples_["samples_of_half"] for samples_ in built_list.values()])
    tags =  np.hstack([np.repeat(samples_["tag"], samples_["samples_of_half"].shape[0]) for samples_ in built_list.values()])
    return samples, tags