import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from typing import List, Optional


def plot_latent_space_3d(
    embeddings: np.ndarray,
    labels: List[str],
    title: str = "Latent Space 3D",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
):
    """
    Projects embeddings into 3D using t-SNE and visualizes them using Plotly.

    Args:
        embeddings: (N, D) array of embeddings.
        labels: List of N labels for coloring.
        title: Title of the plot.
        n_neighbors: t-SNE parameter (complexity).
        min_dist: t-SNE parameter.
        random_state: Seed for reproducibility.
    """
    tsne = TSNE(
        n_components=3,
        random_state=random_state,
        perplexity=min(30, len(embeddings) - 1),
        init="pca",
        learning_rate="auto",
    )
    projections = tsne.fit_transform(embeddings)

    df = pd.DataFrame(projections, columns=["x", "y", "z"])
    df["Label"] = labels

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="Label",
        title=title,
        labels={"Label": "Pain Scale"},
        opacity=0.8,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
        ),
    )

    return fig
