#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : cluster.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 11:45 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 11:45 下午 by shendu.ht  init
"""
import numpy as np


def quality_threshold(x, cutoff=5.0, pbc=False, height=0, width=0):
    """
    Run the complete clustering algorithm in one go and returns the clustered indices as a list.

    Parameters
    ----------
    x: list, np.ndarray
        The input dataset
    cutoff: float
        The clustering cutoff distance.
    pbc: bool
        Activate/Deactivate Periodic Boundary Conditions.
    height: int
        Number of nodes along the first dimension, required for PBC.
    width: int
        Number of nodes along the second dimension, required for PBC.

    Returns
    -------
        list
            A list of lists containing the points indices belonging to each cluster
    """

    clusters = []
    index_list = list(range(len(x)))

    while len(index_list) > 0:
        q_threshold_list = []
        for i in index_list:
            cluster_list = []
            for j in index_list:

                x_i, x_j = np.asarray(x[i]), np.asarray(x[j])
                if pbc is True:
                    offset = 0 if height % 2 == 0 else 0.5
                    height_d = height * 2 / np.sqrt(3) * 3 / 4

                    distance = np.sqrt(np.sum((x_i - x_j) ** 2))
                    right_d = np.sqrt(np.sum((x_i - x_j + np.array([width, 0])) ** 2))
                    bottom_d = np.sqrt(np.sum((x_i - x_j + np.array([offset, height_d])) ** 2))
                    left_d = np.sqrt(np.sum((x_i - x_j + np.array([-width, 0])) ** 2))
                    top_d = np.sqrt(np.sum((x_i - x_j + np.array([-offset, -height_d])) ** 2))
                    bottom_right_d = np.sqrt(np.sum((x_i - x_j + np.array([width + offset, height_d])) ** 2))
                    bottom_left_d = np.sqrt(np.sum((x_i - x_j + np.array([-width + offset, height_d])) ** 2))
                    top_right_d = np.sqrt(np.sum((x_i - x_j + np.array([width - offset, -height_d])) ** 2))
                    top_left_d = np.sqrt(np.sum((x_i - x_j + np.array([-width - offset, -height_d])) ** 2))
                    dist_bmu = np.min([distance, right_d, bottom_d, left_d, top_d,
                                       bottom_right_d, bottom_left_d, top_right_d, top_left_d])
                else:
                    dist_bmu = np.sqrt(np.sum((x_i - x_j) ** 2))

                if dist_bmu <= cutoff:
                    cluster_list.append(j)
            q_threshold_list.append(cluster_list)

        cluster_choose = max(q_threshold_list, key=len)
        clusters.append(cluster_choose)
        for index in cluster_choose:
            index_list.remove(index)

    return clusters


def density_peak():
    return
