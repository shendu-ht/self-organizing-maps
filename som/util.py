#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : util.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 9:24 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 9:24 下午 by shendu.ht  init
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon


def rec_to_hex(x, y):
    """
    Convert Cartesian coordinates to hexagonal tiling coordinates

    Parameters
    ----------
    x: float
        Position along the x-axis of Cartesian coordinates.
    y: float
        Position along the x-axis of Cartesian coordinates.

    Returns
    -------
    list:
        A 2d array containing the coordinates in the new space.

    """
    new_y = y * 2 / np.sqrt(3) * 3 / 4
    new_x = x
    if y % 2:
        new_x += 0.5
    return [new_x, new_y]


def plot_hex(fig, centers, weights):
    """
    Plot an hexagonal grid based on the nodes positions and color the tiles according to their weights.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        The figure on which the hexagonal grid will be plotted.
    centers: list
        Array containing couples of coordinates for each cell to be plotted in the Hexagonal tiling space.
    weights: list
        array containing information on the weights of each cell, to be plotted as colors.

    Returns
    -------
    ax: matplotlib.axes._subplots.AxesSubplot
        The axis on which the hexagonal grid has been plotted.
    """

    ax = fig.add_subplot(111, aspect='equal')

    x_points = [x[0] for x in centers]
    y_points = [x[1] for x in centers]

    if any(isinstance(element, list) for element in weights) and len(weights[0]) == 3:
        for x, y, w in zip(x_points, y_points, weights):
            hexagon = RegularPolygon((x, y), numVertices=6, radius=0.95 / np.sqrt(3), orientation=0, facecolor=w)
            ax.add_patch(hexagon)
    else:
        patches = []
        c_map = plt.get_cmap('viridis')
        for x, y, w in zip(x_points, y_points, weights):
            hexagon = RegularPolygon((x, y), numVertices=6, radius=0.95 / np.sqrt(3), orientation=0, facecolor=c_map(w))
            patches.append(hexagon)

        p = PatchCollection(patches)
        p.set_array(np.array(weights))
        ax.add_collection(p)

    ax.axis('off')
    ax.autoscale_view()
    return ax
