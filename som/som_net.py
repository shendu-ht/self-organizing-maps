#!/usr/local/bin/python3.8
# -*- coding: utf-8 -*-
"""
    Shendu HT
    Copyright (c) 2020-now All Rights Reserved.
    ----------------------------------------------------
    File Name : som_net.py
    Author : shendu.ht
    Email : shendu.ht@outlook.com
    Create Time : 11:07 下午
    Description : description what the main function of this file
    Change Activity :
            version0 : 11:07 下午 by shendu.ht  init
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm, patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import cluster
from sklearn.decomposition import PCA

from som.cluster import quality_threshold
from som.util import rec_to_hex, plot_hex


class SomNet:
    """
    Kohonen Self Organizing Maps Network.
    """

    def __init__(self, height, width, x, load_file=None, pca=False, pbc=False, color_ex=False):
        """
        Initial SOM Network.

        Parameters
        ----------
        height: int
            Number of nodes along the first dimension.
        width: int
            Number of nodes along the second dimension.
        x: Union[list, np.ndarray, pd.Series]
            Dataset. N-dimension
        load_file: str
            (optional) File that contains information to initialise the network weights.
        pca: bool
            Activate/Deactivate Principal Component Analysis to set the initial value of weights.
        pbc: bool
            Activate/Deactivate Periodic Boundary Conditions, only quality threshold clustering works with PBC.
        color_ex: bool
            Activate/Deactivate special workflow if running the colours example.
        """

        self.color_ex = color_ex
        self.pca = pca
        self.pbc = pbc

        self.node_list = []

        # initial input x
        if isinstance(x, list) or isinstance(x, pd.Series):
            x = np.asarray(x)
        elif not isinstance(x, np.ndarray):
            raise ValueError('Input x Only Support (`np.ndarray`, `pd.Series`, `list`, )')
        elif len(x.shape) != 2:
            raise ValueError('Input x should be a matrix, not vector')
        self.x = x

        # initial weights of SOM, randomly or from PCA
        if load_file is None:
            self.height = height
            self.width = width

            min_val, max_val = [], []
            pca_vec = None

            # Whether use PCA to initial weights.
            if self.pca is True:
                pca_ = PCA(n_components=2)
                pca_.fit(self.x)
                pca_vec = pca_.components_
            else:
                for i in range(self.x.shape[1]):
                    min_val.append(np.min(self.x[:, i]))
                    max_val.append(np.max(self.x[:, i]))

            for i in range(self.width):
                for j in range(self.height):
                    self.node_list.append(
                        SomNode(i, j, self.x.shape[1], self.height, self.width, self.pbc, min_val=min_val,
                                max_val=max_val, pca_vec=pca_vec))
        # load weights from file
        else:

            if load_file.endswith('.npy') is False:
                load_file = load_file + '.npy'
            weight_vec = np.load(load_file)

            # use parameters in file
            self.height, self.width, self.pbc = int(weight_vec[0][0]), int(weight_vec[0][1]), bool(weight_vec[0][2])

            # start from 1 because 0 contains info on the size of the network
            count = 1
            for i in range(self.width):
                for j in range(self.height):
                    self.node_list.append(
                        SomNode(i, j, self.x.shape[1], self.height, self.width, self.pbc, weight_vec=weight_vec))
                    count += 1

    def save(self, name='Trained_SomNet', path='./'):
        """
        Saves the network dimensions, the pbc and nodes weights to a file.

        Parameters
        ----------
        name: str
            Name of file where the data will be saved.
        path: str
            Path of file where the data will be saved.
        """

        weight_list = []

        # save som parameters
        som_params = np.zeros(self.node_list[0].weights.size)
        som_params[:3] = [self.height, self.width, int(self.pbc)]
        weight_list.append(weight_list)

        for node in self.node_list:
            weight_list.append(node.weights)
        np.save(os.path.join(path, name), np.asarray(weight_list))

    def update_weight(self, n_iter):
        """
        Update the gaussian sigma.

        Parameters
        ----------
        n_iter: int
            Iteration number
        """
        self.sigma = self.start_sigma * np.exp(-n_iter / self.tau)

    def update_l_rate(self, n_iter):
        """
        Update the learning rate.

        Parameters
        ----------
        n_iter: int
            Iteration number.
        """
        self.l_rate = self.start_l_rate * np.exp(-n_iter / self.epochs)

    def find_bmu(self, vec):
        """
        Find the best matching unit (BMU) for a given vector.

        Parameters
        ----------
        vec: np.ndarray
            The vector to match.

        Returns
        -------
        SomNode
            The best matching unit node.

        """

        # np.float max value
        min_val = np.finfo(np.float).max

        bmu = self.node_list[0]
        for node in self.node_list:
            dist = node.get_distance(vec)
            if dist < min_val:
                min_val = dist
                bmu = node
        return bmu

    def train(self, start_l_rate=0.01, epochs=-1):
        """
        Train the SOM.

        Parameters
        ----------
        start_l_rate: float
            Initial learning rate.
        epochs: int
            Number of training iterations.
            If not selected (or -1) automatically set epochs as 10 times the number of datapoints

        """

        self.start_sigma = max(self.height, self.width) / 2
        self.start_l_rate = start_l_rate

        if epochs == -1:
            epochs = self.x.shape[0] * 10
        self.epochs = epochs
        self.tau = self.epochs / np.log(self.start_sigma)

        for i in range(self.epochs):

            if i % 100 == 0:
                print('\rTraining SOM... {ratio}%'.format(ratio=int(i * 100 / self.epochs)), end=' ')

            self.update_weight(i)
            self.update_l_rate(i)

            vec = self.x[np.random.randint(0, self.x.shape[0]), :].reshape(np.array([self.x.shape[1]]))
            bmu = self.find_bmu(vec=vec)

            for node in self.node_list:
                node.update_weights(vec, self.sigma, self.l_rate, bmu)

        print('\rTraining SOM... done!')

    def nodes_graph(self, column=0, show=False, save=True, path='./.', col_name=None):
        """
        Plot a 2D map with hexagonal nodes and weights values

        Parameters
        ----------
        column: int
            The index of the weight that will be shown as colormap.
        show: bool
            Choose to display the plot.
        save: bool
            Choose to save the plot to a file.
        path: str
            File path to save the plot
        col_name: str
            Name of the column to be shown on the map.
        """

        if col_name is None:
            col_name = str(column)

        centers = [[node.pos[0], node.pos[1]] for node in self.node_list]
        width_point, dpi = 100, 72
        x_inch, y_inch = self.width * width_point / dpi, self.height * width_point / dpi
        fig = plt.figure(figsize=(x_inch, y_inch), dpi=dpi)

        if self.color_ex is True:
            cols = [[np.float(node.weights[0]), np.float(node.weights[1]), np.float(node.weights[2])] for node in
                    self.node_list]
            ax = plot_hex(fig, centers, cols)
            ax.set_title('Node Grid w Color Features', size=80)
            file_name = os.path.join(path, 'nodes_colors.png')

        else:
            cols = [node.weights[column] for node in self.node_list]
            ax = plot_hex(fig, centers, cols)
            ax.set_title('Node Grid w Color Feature ' + col_name, size=80)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.0)
            c_bar = plt.colorbar(ax.collections[0], cax=cax)
            c_bar.set_label(col_name, size=80, labelpad=50)
            c_bar.ax.tick_params(labelsize=60)
            plt.sca(ax)

            file_name = os.path.join(path, 'nodes_colors_{i}.png'.format(i=col_name))

        if save is True:
            plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
        if show is True:
            plt.show()
        plt.clf()

    def diff_graph(self, show=False, save=True, path='./.'):
        """
        Plot a 2D map with nodes and weights difference among neighbouring nodes.

        Parameters
        ----------
        show: bool
            Choose to display the plot.
        save: bool
            Choose to save the plot to a file.
        path: str
            File path to save the plot

        """

        diffs = []
        for node_i in self.node_list:
            diff = 0
            for node_j in self.node_list:
                if node_i != node_j and node_i.get_node_distance(node_j) < 1.001:
                    diff += node_i.get_distance(node_j.weights)
            diffs.append(diff)

        centers = [[node.pos[0], node.pos[1]] for node in self.node_list]
        width_point, dpi = 100, 72
        x_inch, y_inch = self.width * width_point / dpi, self.height * width_point / dpi
        fig = plt.figure(figsize=(x_inch, y_inch), dpi=dpi)
        ax = plot_hex(fig, centers, diffs)
        ax.set_title('Nodes Grid w Weights Difference', size=80)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.0)
        c_bar = plt.colorbar(ax.collections[0], cax=cax)
        c_bar.set_label('Weights Difference', size=80, labelpad=50)
        c_bar.ax.tick_params(labelsize=60)
        plt.sca(ax)
        file_name = os.path.join(path, 'nodes_difference.png')

        if save is True:
            plt.savefig(file_name, bbox_inches='tight', dpi=dpi)
        if show is True:
            plt.show()
        plt.clf()

    def project(self, x_new, column=-1, labels=None, show=False, save=True, path='./.', col_name=None):
        """
        Project the data points of a given array to the 2D space of the SOM by calculating the bmus.

        Parameters
        ----------
        x_new: np.ndarray
            An array containing data points to be mapped.
        column: int
            The index of the weight shown as colormap. If not chosen, the difference map will be used instead.
        labels: list
            The label of x_new.
        show: bool
            Choose to display the plot.
        save: bool
            Choose to save the plot to a file.
        path: str
            File path to save the plot.
        col_name: str
            Name of the column to be shown on the map.

        Returns
        -------
            list
                bmu x,y position for each input array datapoint.
        """

        if col_name is None:
            col_name = str(column)

        label_to_color = {}
        if labels is not None:
            colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                      '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']

            for i in range(len(labels)):
                if labels[i] not in label_to_color:
                    label_to_color[labels[i]] = colors[i % len(colors)]

        bmu_list, cls = [], []
        for i in range(x_new.shape[0]):
            bmu_list.append(self.find_bmu(x_new[i, :]).pos)
            if self.color_ex is True:
                cls.append(x_new[i, :])
            else:
                if labels is not None:
                    cls.append(label_to_color[labels[i]])
                elif column == -1:
                    cls.append('#ffffff')
                else:
                    cls.append(x_new[i, column])

        # Call nodes_graph/diff_graph to first build the 2D map of the nodes
        if self.color_ex is True:
            file_name = os.path.join(path, 'color_projection.png')
            # self.nodes_graph(column, False, False)
            plt.scatter([pos[0] for pos in bmu_list], [pos[1] for pos in bmu_list], color=cls, s=500,
                        edgecolor='#ffffff', linewidth=5, zorder=10)
            plt.title('Data Points Projection', size=80)
        else:
            if column == -1:
                file_name = os.path.join(path, 'projection_difference.png')
                # self.diff_graph(False, False, path)
                plt.scatter([pos[0] - 0.125 + np.random.rand() * 0.25 for pos in bmu_list],
                            [pos[1] - 0.125 + np.random.rand() * 0.25 for pos in bmu_list], c=cls, cmap=cm.viridis,
                            s=400, linewidth=0, zorder=10)
                plt.title('DataPoints Projection on Nodes Difference', size=40)
            else:
                file_name = os.path.join(path, 'projection_{i}.png'.format(i=col_name))
                # self.nodes_graph(column, False, False, col_name=col_name)
                plt.scatter([pos[0] - 0.125 + np.random.rand() * 0.25 for pos in bmu_list],
                            [pos[1] - 0.125 + np.random.rand() * 0.25 for pos in bmu_list], c=cls, cmap=cm.viridis,
                            s=400, edgecolor='#ffffff', linewidth=4, zorder=10)
                plt.title('DataPoints Projection #{i}'.format(i=col_name), size=40)

        if labels is not None:
            recs = []
            for i in label_to_color.keys():
                recs.append(patches.Rectangle((0, 0), 1, 1, fc=label_to_color[i]))
            plt.legend(recs, label_to_color.keys(), loc=0)

        if save is True:
            plt.savefig(file_name, bbox_inches='tight', dpi=72)
        if show is True:
            plt.show()
        plt.clf()
        return [[pos[0], pos[1]] for pos in bmu_list]

    def cluster(self, x_new, c_type='q_threshold', cutoff=5.0, quantile=0.2, num=8, save_cluster=True, filetype='dat',
                show=False, save_plot=True, path='./'):
        """
        Clusters the data in a given array according to the SOM trained map. The clusters can also be plotted.

        Parameters
        ----------
        x_new: np.ndarray
            An array containing data points to be clustered.
        c_type: str
            The type of clustering to be applied, so far only quality threshold (q_threshold)
                algorithm is directly implemented, other algorithms require sklearn.
        cutoff: float
            Cutoff for the quality threshold algorithm. This also doubles as maximum distance of
                two points to be considered in the same cluster with DBSCAN.
        quantile: float
            Quantile used to calculate the bandwidth of the mean shift algorithm.
        num: int
            The number of clusters for K-Means clustering.
        save_cluster: bool
            Choose to save the resulting clusters in a text file.
        filetype: str
            Format of the file where the clusters will be saved (csv or dat).
        show: bool
            Choose to display the plot.
        save_plot: bool
            Choose to save the plot to a file.
        path: str
            File path to save the plot.

        Returns
        -------
            list
                A nested list containing the clusters with indexes of the input array points.
        """

        bmu_list = self.project(x_new, show=False, save=False)

        if c_type == 'q_threshold':

            """ Cluster according to the quality threshold algorithm (slow!). """

            clusters = quality_threshold(bmu_list, cutoff=cutoff, pbc=self.pbc, height=self.height, width=self.width)

        # elif c_type == 'density_peak':
        #
        #     """ Cluster according to the density peak algorithm. """
        #
        #     clusters = density_peak()

        elif c_type in ['mean_shift', 'dbscan', 'k_means']:

            """ Cluster according to algorithms implemented in sklearn. """

            if c_type == 'mean_shift':
                bandwidth = cluster.estimate_bandwidth(np.asarray(bmu_list), quantile=quantile, n_samples=500)
                c = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(bmu_list)

            elif c_type == 'dbscan':
                c = cluster.DBSCAN(eps=cutoff, min_samples=5).fit(bmu_list)

            elif c_type == 'k_means':
                c = cluster.KMeans(n_clusters=num).fit(bmu_list)

            else:
                raise ValueError('Unknown clustering algorithm, {s}'.format(s=c_type))

            c_labels = c.labels_

            cluster_dict = {}
            for i in range(len(c_labels)):
                if c_labels[i] not in cluster_dict:
                    cluster_dict[c_labels[i]] = []
                cluster_dict[c_labels[i]].append(i)

            clusters = list(cluster_dict.values())

        else:
            raise ValueError('Unknown clustering algorithm, {s}'.format(s=c_type))

        if save_cluster is True:
            file = open(os.path.join(path, c_type + '_clusters.' + filetype), 'w')
            if filetype == 'csv':
                separator = ','
            else:
                separator = ' '

            for line in clusters:
                file.write(separator.join(map(str, line)) + '\n')
            file.close()

        np.random.seed(0)
        file_name = os.path.join(path, c_type + '_clusters.png')

        fig, ax = plt.subplots()

        for i in range(len(clusters)):
            rand_color = '#%06x' % np.random.randint(0, 0xFFFFFF)
            xc = [bmu_list[index][0] for index in clusters[i]]
            yc = [self.height - bmu_list[index][1] for index in clusters[i]]
            ax.scatter(xc, yc, color=rand_color, label='cluster_{i}'.format(i=i))

        plt.gca().invert_yaxis()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.set_title('Clusters')
        ax.axis('off')

        if save_plot is True:
            plt.savefig(file_name, bbox_inches='tight', dpi=600)
        if show is True:
            plt.show()
        return clusters


class SomNode:
    """
    Single node class for Kohonen Self Organizing Maps Network.
    """

    def __init__(self, x, y, weights, height, width, pbc, min_val=None, max_val=None, pca_vec=None, weight_vec=None):
        """
        Initialize the SOM node.

        Parameters
        ----------
        x: int
            Position along the first network dimension.
        y: int
            Position along the second network dimension.
        weights: int
            Length of the weights vector.
        height: int
            Network height, needed for periodic boundary conditions (PBC)
        width: int
            Network width, needed for periodic boundary conditions (PBC)
        pbc: bool
            Activate/deactivate periodic boundary conditions.
        min_val: list
            Minimum values for the weights found in the data
        max_val: list
            Maximum values for the weights found in the data
        pca_vec: Union[list, np.ndarray, pd.Series]
            Containing the two PCA vectors.
        weight_vec: Union[list, np.ndarray, pd.Series]
            Containing the weights to give to the node if a file was loaded.
        """

        self.pbc = pbc
        self.pos = np.asarray(rec_to_hex(x, y))

        self.height = height
        self.width = width

        weights_list = []
        if weight_vec is None and pca_vec is None:
            if min_val is None or max_val is None:
                raise ValueError('Min/Max Val is None')

            # select randomly in the space spanned by the data
            for i in range(weights):
                weights_list.append(np.random.random() * (max_val[i] - min_val[i]) + min_val[i])
        elif weight_vec is None and pca_vec is not None:
            # select uniformly in the space spanned by the PCA vectors
            pca_weight = (x - self.width / 2) * 2.0 / self.width * pca_vec[0] + (
                    y - self.height / 2) * 2.0 / self.height * pca_vec[1]
            weights_list = [pca_weight for _ in range(weights)]
        else:
            if len(weight_vec) < weights:
                raise ValueError('the length of weight vector is less than input weights')
            for i in range(weights):
                weights_list.append(weight_vec[i])

        self.weights = np.asarray(weights_list)

    def get_distance(self, vec):
        """
        Calculate the distance between the weights vector of the node and a given vector.

        Parameters
        ----------
        vec: np.ndarray
            The vector from which the distance is calculated.

        Returns
        -------
        float
            The distance between the two weight vectors.

        """
        if vec.size != self.weights.size:
            raise ValueError('vec dimension != node dimension')
        return np.sqrt(np.sum((self.weights - vec) ** 2))

    def get_node_distance(self, node):
        """
        Calculate the distance within the network between the node and another node.

        Parameters
        ----------
        node: SomNode
            The node from which the distance is calculated.

        Returns
        -------
        float
            The distance between the two nodes.

        """

        if self.pbc is True:
            offset = 0 if self.height % 2 == 0 else 0.5

            # Hexagonal Periodic Boundary Conditions
            height_d = self.height * 2 / np.sqrt(3) * 3 / 4
            distance = np.sqrt(np.sum((self.pos - node.pos) ** 2))
            right_d = np.sqrt(np.sum((self.pos - node.pos + np.array([self.width, 0])) ** 2))
            bottom_d = np.sqrt(np.sum((self.pos - node.pos + np.array([offset, height_d])) ** 2))
            left_d = np.sqrt(np.sum((self.pos - node.pos + np.array([-self.width, 0])) ** 2))
            top_d = np.sqrt(np.sum((self.pos - node.pos + np.array([-offset, -height_d])) ** 2))
            bottom_right_d = np.sqrt(np.sum((self.pos - node.pos + np.array([self.width + offset, height_d])) ** 2))
            bottom_left_d = np.sqrt(np.sum((self.pos - node.pos + np.array([-self.width + offset, height_d])) ** 2))
            top_right_d = np.sqrt(np.sum((self.pos - node.pos + np.array([self.width - offset, -height_d])) ** 2))
            top_left_d = np.sqrt(np.sum((self.pos - node.pos + np.array([-self.width - offset, -height_d])) ** 2))
            return np.min(
                [distance, right_d, bottom_d, left_d, top_d, bottom_right_d, bottom_left_d, top_right_d, top_left_d])
        else:
            return np.sqrt(np.sum((self.pos - node.pos) ** 2))

    def update_weights(self, vec, sigma, l_rate, bmu):
        """
        Update the node Weights.

        Parameters
        ----------
        vec: np.ndarray
            A weights vector whose distance drives the direction of the update.
        sigma: float
            The updated gaussian sigma.
        l_rate: float
            The updated learning rate.
        bmu: SomNode
            The best matching unit.
        """
        dist = self.get_node_distance(bmu)
        gauss = np.exp(-dist ** 2 / (2 * sigma ** 2))

        self.weights = self.weights - gauss * l_rate * (self.weights - vec)
