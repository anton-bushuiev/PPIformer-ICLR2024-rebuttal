import numpy as np
import pandas as pd
from torchmetrics import ConfusionMatrix, ROC
import plotly.graph_objects as go


from ppiformer.utils.bio import BASE_AMINO_ACIDS, BASE_AMINO_ACIDS_GROUPED, class_to_amino_acid
from ppiref.utils.ppipath import path_to_ppi_id


# TODO Move `.compute().cpu().detach().numpy()` logic in all functions to model classes
def plot_confusion_matrix(confmat: ConfusionMatrix, log_scale: bool = False):
    df = pd.DataFrame(
        confmat.compute().cpu().detach().numpy(),
        index=BASE_AMINO_ACIDS,
        columns=BASE_AMINO_ACIDS
    )
    df = df.loc[BASE_AMINO_ACIDS_GROUPED, BASE_AMINO_ACIDS_GROUPED]

    with np.errstate(divide='ignore'):
        z = df.values if not log_scale else np.log(df.values)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            y=df.index,
            x=df.columns,
            colorscale='Aggrnyl'
        )
    )
    fig.update_layout(yaxis = dict(scaleanchor = 'x'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(autosize=False, width=500, height=500)
    
    return fig

def plot_roc_curve(tpr, fpr, threshold):

    fig = go.Figure()
    # Add ROC curve trace
    fig.add_trace(go.Scatter(x=fpr, y=tpr,
                             mode='lines',
                             name=f'ROC Curve',
                             line=dict(color='blue', width=2)))

    # Add diagonal reference line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random Guessing',
                             line=dict(color='black', dash='dash')))
    fig.update_layout(autosize=False, width=500, height=500)
    
    return fig



# NOTE: Works only for single-point masking
def plot_classification_heatmap_example(data, y_proba):
    # Construct dataframe
    nodes = np.hstack(data.node_id)[~data.node_mask.cpu().detach()]
    pdb_ids = [path.split('/')[-1].rsplit('.', 1)[0] for path in data.path]
    idx = [':'.join([pdb, n]) for n, pdb in zip(nodes, pdb_ids)]
    df = pd.DataFrame(
        y_proba.cpu().detach().numpy(),
        columns=BASE_AMINO_ACIDS,
        index=idx
    )
    df = df[BASE_AMINO_ACIDS_GROUPED]

    # Define text
    wts = data.y[~data.node_mask]
    text = np.full((len(wts), 20), '')
    for n, a in zip(np.arange(len(wts)), wts):
        text[n, a] = class_to_amino_acid(a)

    # Plot
    fig = go.Figure(
        data=go.Heatmap(
            z=df.values,
            y=df.index,
            x=df.columns,
            text=text,
            texttemplate='%{text}',
            textfont={'size': 10}
        )
    )
    fig.update_layout(yaxis = dict(scaleanchor = 'x'))
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(autosize=False, width=500, height=500)

    return fig


def plot_ddg_scatter(ddg_pred, ddg_true, batch):
    ddg_true = ddg_true.cpu().detach().numpy()
    ddg_pred = ddg_pred.cpu().detach().numpy()
    batch = batch.cpu().detach().numpy()

    # TODO pass sample_paths for text=ppi_ids
    # sample_ppi_ids = list(map(path_to_ppi_id, sample_paths))
    # ppi_ids = np.array(sample_ppi_ids)[batch]

    fig = go.Figure(
        data=go.Scatter(
            x=ddg_true,
            y=ddg_pred,
            mode='markers',
            marker_color=batch
        )
    )
    fig.update_layout(
        xaxis_title='True ddG',
        yaxis_title='Predicted ddG'
    )
    return fig
