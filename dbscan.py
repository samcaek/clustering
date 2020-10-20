import logging
import math
import random
import sys
from multiprocessing.pool import Pool
from time import time
from typing import List, Tuple
import argparse

import numpy as np

from data import data_set_names, preprocessed_data_sets
from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet
from utils import logging_util
from sklearn import metrics
import matplotlib.pyplot as plt

sys.setrecursionlimit(10000)

class DBScan():
    """
    Class that implements the K-Means algorithm.
    """

    def __init__(self, epsilon=0.05, minpts=5):
        """
        Initializes a KMeans object.

        :param k: The amount of centers to find.
        :param improvement_threshold: How low the change in the centers must be before the algorithm stops.
        """
        self.k = 0
        self.empty_cluster = False
        self.epsilon = epsilon
        self.id = {}
        self.minpts = minpts
        self.purity = -100
        self.sil_coef = -100


        

    def dbscan(self, data_set: DataSet) -> DataSet:
        """
        Runs the K Means algorithm on the data set to find centers. 
        See: https://dataminingbook.info/book_html/chap15/book-watermark.html

        :param data_set: The data used to find the centers.
        :return: The data set of centers.
        """
        class_labels = data_set.get_output_values()
        logging.info(f'\nRunning {self} on dataset:')
        data_set = np.array(data_set)
        core = []
        # Find the core points
        for i in range(len(data_set)):
            n_x = e_neighboorhood(data_set, data_set[i], self.epsilon)
            self.id[i] = None
            if len(n_x) >= self.minpts:
                core.append(i)


        k = 0
        print("core is %s" % core)
        # For each core such that ... line7
        for i in core:
            if self.id[i]:
                continue
            k += 1
            self.k = k
            self.id[i] = k
            self.density_connected(data_set, i, k, core)

        # line 11-14
        C = {}
        print("k is %s" % k)
        for i in range(1,k+1):
            for x in range(len(data_set)):
                if self.id.get(x) == i:
                    C[i] = C.get(i,[]) + [x]

        noise = [i for i in range(len(data_set)) if not self.id.get(i)]
        border = [i for i in range(len(data_set)) if i not in core and i not in noise]

        bool_array = self.id_to_bool_array(data_set)
        self.silhoutte_coef(bool_array, data_set)

        self.calc_purity(bool_array, data_set, class_labels)


        return C, core, border, noise

    def assign_border_points(self, border_points, data_set):
        for i in border_points:
            most_common_list = []
            for y in e_neighboorhood(data_set, data_set[i], self.epsilon):
                if self.id[y]:
                    most_common_list.append(self.id[y])

            most_common = max(set(most_common_list), key=most_common_list.count)

            self.id[i] = most_common

    def density_connected(self, data_set,  i, k, core):
        for y in e_neighboorhood(data_set, data_set[i], self.epsilon):
            if self.id[y]:
                continue
            self.id[y] = k
            if y in core:
                self.density_connected(data_set, y, k, core)

    def id_to_bool_array(self, data_set):
        
        bool_array = np.zeros((self.k, len(data_set)), dtype='bool')
        for i in range(len(data_set)):
            if self.id.get(i):
                bool_array[self.id.get(i)-1, i] = True

        return bool_array

    def __repr__(self):
        return f'DBSCan(k={self.k}, epislon={self.epsilon}, minpts={self.minpts})'

    def silhoutte_coef(self, bool_array: np.ndarray, data_set: np.ndarray):
        """
        Calculates the silhoutte coef for each point in the data set
        See: https://en.wikipedia.org/wiki/Silhouette_(clustering)

        :param bool_array: The boolean array used to index the data_set to get the clusters.
        :param data_set: The data used to find the centers.
        :return: None
        """
        empty_count = 0

        coef_list = []
        for i in range(self.k):
            records_in_cluster = data_set[bool_array[i]]
            if len(records_in_cluster) == 0:
                self.empty_cluster = True
                empty_count += 1
            elif not self.empty_cluster:
                a = 0
                for i1, r1 in enumerate(records_in_cluster):
                    total_a_dist = 0
                    for i2, r2 in enumerate(records_in_cluster):
                        # Get intra-cluster distance for point r1
                        if i2 == i1:
                            continue
                        else:
                            total_a_dist += (np.sum((r1 - r2) ** 2) ** (1/2))
                    a = total_a_dist/len(records_in_cluster)

                    min_b_dist = 999999999999999
                    for j in range(self.k):
                        # loop through every outer cluster and find min distance cluster
                        if i == j:
                            continue
                        records_in_cluster_j = data_set[bool_array[j]]
                        b = sum([np.sum((r1 - r3) ** 2) ** (1/2) for r3 in records_in_cluster_j]) / len(records_in_cluster_j)

                        if b < min_b_dist:
                            min_b_dist = b

                    # return min distance to nearest cluster
                    b = min_b_dist
                    sil_score = (b-a)/max(b,a)
                    coef_list.append(sil_score)
        
        mean_sil_score = sum(coef_list)/len(coef_list)
        print("mean silhoutte score is %s" % mean_sil_score)
        self.sil_coef = mean_sil_score
        return mean_sil_score

    def calc_purity(self, bool_array: np.ndarray, data_set: np.ndarray, class_labels): 
        # Adapted from https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
        y_true = class_labels
        y_pred = []
        for c,col in enumerate(bool_array[0]):
            added = False
            for r, row in enumerate(bool_array):
                if bool_array[r][c] == True:
                    y_pred.append(r)
                    added = True
            if not added:
                y_pred.append(-1)
        
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        print(contingency_matrix)
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
        self.purity = purity
        return purity

def e_neighboorhood(dataset, point, epsilon):
    return [i for i in range(len(dataset)) if np.sum((dataset[i] - point) ** 2) ** (1/2) < epsilon and not (point==dataset[i]).all()]



def main(data_sets_to_process=None):
    parser = argparse.ArgumentParser(description='Dbscan')

    parser.add_argument('--epsilon', action="store", default=0.1)
    parser.add_argument('--minpts', action="store", default=4)
    parser.add_argument('--dataset', action="store", default="iris")


    args = parser.parse_args()
    epsilon = float(args.epsilon)
    minpts = int(args.minpts)
    random.seed(123)
    np.random.seed(2)
    #logging_util.start_logging()

    if data_sets_to_process is not None:
        pass
    elif args.dataset:
        if args.dataset == "synthetic":
            data_sets_to_process = ["synthetic"]
        else:
            data_sets_to_process = [args.dataset]
    else:
        data_sets_to_process = data_set_names

    for name in data_sets_to_process:
        preprocessed_data_set = preprocessed_data_sets[name]
        dbscan = DBScan(epsilon=epsilon, minpts=minpts)

        C, core, border, noise = dbscan.dbscan(preprocessed_data_set)
        #dbscan_data_sets[name] = C

        print("")
        print("C is:")
        print(C)

        print("")
        print("core is \n %s" % core)

        print("")
        print("noise is:")
        print(noise)

        print("")
        print("border is:")
        print(border)

        # for id in dbscan.id.keys():
        #     print("id is %s -- cluster is %s " % (id,dbscan.id[id]) )

    print("sil_coef is %s" % dbscan.sil_coef)
    print("purity is %s" % dbscan.purity)
    data_set = np.array(preprocessed_data_set)

    x = [row[0] for row in data_set]
    y = [row[1] for row in data_set]
    colors_true = preprocessed_data_set.get_output_values()

    colors_pred = [dbscan.id.get(i) for i in range(len(data_set))]
    colors_pred = [-1 if i == "None" or not i else i for i in colors_pred]

    plt.scatter(x, y, c=colors_true, alpha=0.5,s=3)
    plt.suptitle('%s Ground-truth Data' % name.capitalize(), fontsize=20)
    plt.show()

    plt.scatter(x, y, c=colors_pred, alpha=0.5,s=3)
    plt.suptitle('DBScan labels (e=%s,minpts=%s)' % (dbscan.epsilon, dbscan.minpts), fontsize=20)
    plt.xlabel("Mean Sillhoutte: %s -- Purity: %s" % (round(dbscan.sil_coef, 3), round(dbscan.purity,3)))

    plt.show()


if __name__ == '__main__':
    main()
