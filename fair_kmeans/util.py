from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def print_clustering(
    data: np.ndarray,
    centers: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    label_colors: Optional[List[str]] = None,
) -> None:
    k = len(centers)
    if label_colors is None:
        label_colors = list(mcolors.CSS4_COLORS.values())

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Only for two colors
    cmap = ListedColormap(["blue", "red"])

    ax1.scatter(
        data[:, 0],
        data[:, 1],
        c=colors,
        cmap=cmap,
    )
    for i, c in enumerate(label_colors[:k]):
        for ax in [ax1, ax2]:
            ax.scatter(
                centers[i, 0],
                centers[i, 1],
                marker="x",
                c=c,
            )

        ax2.scatter(
            data[labels == i][:, 0],
            data[labels == i][:, 1],
            c=c,
        )
