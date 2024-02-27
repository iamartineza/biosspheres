from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def template_3d_surface_plot(
    vector: np.ndarray,
    normalized_surface_field: np.ndarray,
    vmin: float,
    vmax: float,
    x_label: str,
    y_label: str,
    z_label: str,
    title: str,
    str_color_bar: str,
    color,
) -> None:
    """

    Parameters
    ----------
    vector: np.ndarray
        Three dimensions, the first one has length 3.
    normalized_surface_field: np.ndarray
        Two dimensions, their lengths match the lengths of vector[0,:,:]
    vmin: float
        Minimal value of the surface field (before normalization).
    vmax: float
        Maximal value of the surface field (before normalization).
    x_label: str
    y_label: str
    z_label: str
    title: str
        Title of the plot.
    str_color_bar: str
        Label of the color bar.
    color: a color map
        For the surface colors.

    Returns
    -------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        vector[0, :, :],
        vector[1, :, :],
        vector[2, :, :],
        rstride=1,
        cstride=1,
        facecolors=color(normalized_surface_field),
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.colorbar(
        cm.ScalarMappable(
            norm=colors.Normalize(vmin=vmin, vmax=vmax),
            cmap=color,
        ),
        ax=ax,
        label=str_color_bar,
    )
    pass


def draw_cut_with_call(
    cut: int,
    center: np.ndarray,
    horizontal: float,
    vertical: float,
    inter_horizontal: int,
    inter_vertical: int,
    colorbar_lab: str,
    title: str,
    u_rf: Callable[[np.ndarray], float],
    colorbar_or: str = "vertical",
    with_plot: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], str, str]:
    x1 = np.linspace(-horizontal / 2, horizontal / 2, inter_horizontal)
    y1 = np.linspace(-vertical / 2, vertical / 2, inter_vertical)
    extent = []
    data_for_plotting = np.zeros((len(y1), len(x1)))
    if cut == 3:
        z = 0.0 + center[2]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                y = y1[jj] + center[1]
                vector = np.asarray([x, y, z])
                data_for_plotting[jj, ii] = u_rf(vector)
            pass
        pass
        extent = [
            -horizontal / 2 + center[0],
            horizontal / 2 + center[0],
            -vertical / 2 + center[1],
            vertical / 2 + center[1],
        ]
        x_label = "$x$"
        y_label = "$y$"
    elif cut == 2:
        y = center[1]
        for ii in np.arange(0, len(x1)):
            x = x1[ii] + center[0]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                vector = np.asarray([x, y, z])
                data_for_plotting[jj, ii] = u_rf(vector)
            pass
        pass
        extent = [
            -horizontal / 2 + center[0],
            horizontal / 2 + center[0],
            -vertical / 2 + center[2],
            vertical / 2 + center[2],
        ]
        x_label = "$x$"
        y_label = "$z$"
    elif cut == 1:
        x = center[0]
        for ii in np.arange(0, len(x1)):
            y = x1[ii] + center[1]
            for jj in np.arange(0, len(y1)):
                z = y1[jj] + center[2]
                vector = np.asarray([x, y, z])
                data_for_plotting[jj, ii] = u_rf(vector)
            pass
        pass
        extent = [
            -horizontal / 2 + center[1],
            horizontal / 2 + center[1],
            -vertical / 2 + center[2],
            vertical / 2 + center[2],
        ]
        x_label = "$y$"
        y_label = "$z$"
    pass
    if with_plot:
        plt.figure()
        plt.imshow(
            data_for_plotting,
            origin="lower",
            extent=extent,
            norm=colors.CenteredNorm(),
        )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.colorbar(label=colorbar_lab, orientation=colorbar_or)
        plt.title(title)
    pass
    return x1, y1, data_for_plotting, extent, x_label, y_label
