import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# def create_histograms_2d_array_v0(array):
#     num_cols = array.shape[1]
#     fig = make_subplots(rows=int(np.ceil(np.sqrt(num_cols))),
#                         cols=int(np.ceil(np.sqrt(num_cols)))
#                         )
#     for col_index in range(num_cols):
#         column_data = array[:, col_index]
#         unique_values, unique_counts = np.unique(column_data, return_counts=True)
#
#         fig.add_trace(go.Bar(x=unique_values, y=unique_counts, name=f'Column {col_index}'),
#                       row = int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
#                       col = int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
#                       )
#
#     fig.show(renderer='browser') # lookup renderer
#         # todo:
#         #   only use as many bins as unique values so no empty bins are shown -> put the values in order.
#         #   if plotly doesn't allow this, map the values to a range of integers
#         #   and plot the histogram with the integer values
#         #   use annotations only upon hover and not on the plot
#         #   for the annotations to be correct use the mapping from the previous step
#         #   put all histograms on the same plot
#
#          # https://www.analyticsvidhya.com/blog/2021/10/interactive-plots-in-python-with-plotly-a-complete-guide/
#         # https://www.datacamp.com/tutorial/create-histogram-plotly
#
#
#         # https://plotly.com/python/renderers/
#         # https://plotly.com/python/histograms/
#         # https://plotly.com/python/figure-labels/
#

# def create_histograms_2d_array_v1(array):
#     num_cols = array.shape[1]
#     fig = make_subplots(rows=int(np.ceil(np.sqrt(num_cols))),
#                         cols=int(np.ceil(np.sqrt(num_cols)))
#                         )
#     for col_index in range(num_cols):
#         column_data = array[:, col_index]
#         unique_values, unique_counts = np.unique(column_data, return_counts=True)
#
#         # Filter out bins with zero counts
#         non_zero_indices = np.where(unique_counts != 0)
#         unique_values = unique_values[non_zero_indices]
#         unique_counts = unique_counts[non_zero_indices]
#
#         fig.add_trace(go.Bar(x=unique_values, y=unique_counts, name=f'Column {col_index}'),
#                       row = int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
#                       col = int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
#                       )
#
#     fig.show(renderer='browser')

def create_histograms_2d_array_v2(array):
    num_cols = array.shape[1]
    fig = make_subplots(rows=int(np.ceil(np.sqrt(num_cols))),
                        cols=int(np.ceil(np.sqrt(num_cols)))
                        )
    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, unique_counts = np.unique(column_data, return_counts=True)
        num_unique_values = len(np.unique(column_data))

        # Filter out bins with zero counts
        # non_zero_indices = np.where(unique_counts != 0)
        # unique_values = unique_values[non_zero_indices]
        # unique_counts = unique_counts[non_zero_indices]
        #
        # fig.add_trace(go.Bar(x=unique_values, y=unique_counts, name=f'Column {col_index}'),
        #               row = int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
        #               col = int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
        #               )
        fig.add_trace(go.Histogram(x=column_data, nbinsx=num_unique_values,
                                   histfunc="count",
                                   name=f'Column {col_index}'),
                      row=int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
                      col=int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
                      )

    fig.show(renderer='browser')

def create_histograms_2d_array_v3(array):
    num_cols = array.shape[1]
    fig = make_subplots(rows=int(np.ceil(np.sqrt(num_cols))),
                        cols=int(np.ceil(np.sqrt(num_cols)))
                        )
    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values = np.unique(column_data)

        # Create custom bins
        bins = np.concatenate((unique_values, [np.max(unique_values) + 1])) - 0.5

        fig.add_trace(go.Histogram(x=column_data, xbins={'start': bins[0], 'end': bins[-1], 'size': 1},
                                   histfunc="count",
                                   name=f'Column {col_index}'),
                      row=int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
                      col=int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
                      )

    fig.show(renderer='browser')

def create_histograms_2d_array_v4_almost(array):
    num_cols = array.shape[1]
    fig = make_subplots(rows=int(np.ceil(np.sqrt(num_cols))),
                        cols=int(np.ceil(np.sqrt(num_cols)))
                        )
    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, unique_counts = np.unique(column_data, return_counts=True)

        # Create a mapping from unique values to indices
        index_to_value = dict(enumerate(unique_values))
        print("Column{}: ".format(col_index))
        print(index_to_value)
        # Convert unique values to indices
        indices = list(index_to_value.keys())

        fig.add_trace(go.Histogram(x=indices, y=unique_counts, name=f'Column {col_index}'),
                      row=int(col_index // np.ceil(np.sqrt(num_cols)) + 1),
                      col=int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
                      )

        # Set the tick labels on the x-axis to be the unique values
        fig.update_xaxes(tickvals=list(index_to_value.keys()),
                         ticktext=list(index_to_value.values()))

    fig.show(renderer='browser')

def create_histograms_2d_array(array, param_names, model_name, paper = False):
# def create_histograms_2d_array(array, param_names = "aa" * 13, model_name = "test", paper = False):
    num_cols = array.shape[1]
    # param_names = [f"col {i}" for i in range(num_cols)] # TODO remove me
    rows_cols_number = int(np.ceil(np.sqrt(num_cols)))
    fig = make_subplots(rows=rows_cols_number,
                        cols=rows_cols_number
                        )
    for col_index in range(num_cols):
        column_data = array[:, col_index]
        unique_values, unique_counts = np.unique(column_data, return_counts=True)

        # Create a mapping from unique values to indices
        index_to_value = dict(enumerate(unique_values))
        # Convert unique values to indices
        indices = list(index_to_value.keys())

        print("Column{}: ".format(col_index))
        print(indices)
        print(unique_counts)
        print(list(index_to_value.values()))
        cr_row = int(col_index // np.ceil(np.sqrt(num_cols)) + 1)
        cr_col = int(col_index % np.ceil(np.sqrt(num_cols)) + 1)
        print(cr_col, cr_row)
        fig.add_trace(go.Bar(x=indices, y=unique_counts, name=f'Column {col_index}'),
                      row=cr_row,
                      col=cr_col
                      )

        # Set the tick labels on the x-axis to be the unique values
        fig.update_xaxes(tickvals=indices,
                         ticktext=list(index_to_value.values()),
                         showticklabels=False,
                         row=cr_row,
                         col=cr_col
                         )
        if paper:
            fig.add_annotation(
                text=f'Hist of model {model_name} \n Params {param_names[col_index]}',
                xref='paper',
                yref='paper',
                # xref=f'x{col_index + 1}',
                # yref=f'y{col_index + 1}',
                showarrow=False,
                font=dict(size=10),
                xanchor='left',
                yanchor='top',
                # x=(cr_col - 1) / rows_cols_number, # either calculate these based on the normalized coords with xref and yref paper
                # y=cr_row / rows_cols_number,# or use the col_index refs AND calculate relative position based on the values in the histogram
                x= (cr_col - 1) / rows_cols_number,
                y=1 - ((cr_row - 1) / rows_cols_number), # (0,0) of paper is bottom left not top left
                yshift=10
            )
        else:
            fig.add_annotation(
                text=f'Hist of model {model_name} \n Params {param_names[col_index]}',
                xref=f'x{col_index + 1}',
                yref=f'y{col_index + 1}',
                showarrow=False,
                font=dict(size=10),
                xanchor='left',
                yanchor='bottom',
                x=min(indices),  # calculate these based on the indices and unique_counts lists
                y=max(unique_counts), # + 0.1 * max(unique_counts),
                # calculate these based on the indices and unique_counts lists
                yshift=10
            )





    fig.show(renderer='browser')

data = np.array([
    [1, 2, 3, 1, 5, 7, 7, 3, 1, 5, 3, 1, 5],
    [4, 5, 6, 1, 1, 4, 4, 6, 1, 1, 6, 1, 1],
    [1, 2, 3, 1, 5, 6, 6, 3, 1, 5, 3, 1, 5],
    [4, 4, 6, 1, 2, 7, 7, 6, 1, 2, 6, 1, 2],
    [1, 51, 61, 41, 42, 47, 47, 46, 41, 42, 46, 41, 35],
    [1, 5, 6, 1, 2, 7, 7, 6, 1, 2, 6, 1, 2],

])
# Example usage

create_histograms_2d_array(data)

#

# title: Plotly not showing histogram but no error message

"""
Hi, I'm trying to create an interactive histogram with plotly, but fig.show() does not open up my visualization. 
I have the following plotly version installed: '5.19.0'

** Python version: ** Python 3.10.13 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:24:38) [MSC v.1916 64 bit (AMD64)] on win32 
IDE: Pycharm Community. 
**System:** Laptop with Windows 11 Education, 64 bit, Version 23H2, OS build 22631.3447

I'm trying to run the ** following code **(both in python console and in a script) but no figure shows (and I get no error message): 

```import plotly.graph_objects as go
import numpy as np
np.random.seed(1)
x = np.random.randn(500)
fig = go.Figure(data=[go.Histogram(x=x)])
fig.show()```

The env I want to use plotly on has the following packages installed: https://pastebin.com/3M07ULae
Also, the TensorFlow 2.10 runs just fine with my RTX 3050 Ti GPU without requiring WSL.

Any ideas on how I could fix this on my Py3.10 TF 2.10 env?
"""

"""
**I tried** installing plotly in another conda env and it displayed the histogram just fine in a browser tab. 
This env has the following packages: https://pastebin.com/eSjYw5p2
"""

# tags: python plotly visualization data-visualization data-science plots
