import logging
import math
import random
import sys
from multiprocessing.pool import Pool
from time import time
from typing import List, Tuple
import argparse
import numpy as np

from data import data_set_names, preprocessed_data_sets, k_means_data_sets
from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet
from utils import logging_util
from sklearn import metrics
import matplotlib.pyplot as plt

class KMeans():
    """
    Class that implements the K-Means algorithm.
    """

    def __init__(self, k: int, iterations=3000, improvement_threshold=0.00001):
        """
        Initializes a KMeans object.

        :param k: The amount of centers to find.
        :param improvement_threshold: How low the change in the centers must be before the algorithm stops.
        """
        self.iterations = iterations
        self.k = k
        self.improvement_threshold = improvement_threshold
        self.empty_cluster = False
        self.sil_coef = -100
        self.purity = -100
        self.y_pred = []
        self.id = {}

    def get_pred(self):
        return [self.id[i] for i in range(len(self.id.keys()))]

    def generate_random_assignment(self, data_set):
        """
        Randomly assigns records in the data set to a cluster.

        :param data_set: The data set to assign.
        :return: The array of bools that indexes the data set.
        """
        bool_array = np.zeros((int(self.k), len(data_set)), dtype='bool')
        cluster_index = 0
        np.random.shuffle(data_set)
        for i in range(len(data_set)):
            bool_array[cluster_index, i] = True
            cluster_index += 1
            if cluster_index == int(self.k):
                cluster_index = 0
        return bool_array

    def center_assignment(self, centers, data_set):
        """
        Assigns the records in the data set to a cluster based on the closest center.

        :param centers: The k means centers.
        :param data_set: The data set to assign.
        :return: The array of bools that indexes the data set.
        """
        bool_array = np.zeros((int(self.k), len(data_set)), dtype='bool')

        for i in range(len(data_set)):
            record = data_set[i]
            distances = np.sum((centers - record) ** 2, axis=1) ** (0.5)
            center_index = np.argmin(distances)
            bool_array[center_index, i] = True
            self.id[i] = center_index

        return bool_array

    def move_centers(self, bool_array: np.ndarray, data_set: np.ndarray, centers: np.ndarray):
        """
        Moves the centers to the means of the clusters.

        :param bool_array: The boolean array used to index the data_set to get the clusters.
        :param data_set: The data used to find the centers.
        :param centers: The centers to be moved.
        :return: None
        """
        empty_count = 0
        for i in range(int(self.k)):
            records_in_cluster = data_set[bool_array[i]]
            if len(records_in_cluster) == 0:
                self.empty_cluster = True
                empty_count += 1
            elif not self.empty_cluster:
                mean = np.mean(records_in_cluster, axis=0)
                centers[i] = mean

        #self.k -= change
        logging.info(self.k)

    def produce(self, data_set: DataSet) -> DataSet:
        """
        Runs the K Means algorithm on the data set to find centers.
        See https://dataminingbook.info/book_html/chap13/book-watermark.html

        :param data_set: The data used to find the centers.
        :return: The data set of centers.
        """
        logging.info(f'\nRunning {self} on dataset:')
        logging.info(data_set.summary())
        class_labels = data_set.get_output_values()
        data_set = np.array(data_set)
        centers = data_set.copy()[:self.k]
        bool_array = self.generate_random_assignment(data_set)
        self.move_centers(bool_array, data_set, centers)
        for i in range(self.iterations):
            bool_array = self.center_assignment(centers, data_set)
            previous_centers = centers.copy()
            self.move_centers(bool_array, data_set, centers)
            max_change = np.max(np.sum(np.absolute(centers - previous_centers), axis=1))
            #print("max_change is %s -- iteration is %s" % (max_change, i))

            if self.empty_cluster:
                self.empty_cluster = False
                bool_array = self.generate_random_assignment(data_set)
                centers = np.zeros((int(self.k), data_set.shape[1]))
                self.move_centers(bool_array, data_set, centers)
                continue

            if max_change < self.improvement_threshold and i > 1000:
                self.silhoutte_coef(bool_array, data_set, centers)
                break


            if i % 100 == 0:
                logging.info(round(i / self.iterations, 2) * 100, '%')

        self.silhoutte_coef(bool_array, data_set, centers)
        self.calc_purity(bool_array, data_set, class_labels)

        centers_set = DataSet(record_ndarray=centers)
        logging.info('\nK-Means centers:')
        logging.info(centers_set.summary())
        return centers_set

    def __repr__(self):
        return f'KMeans(k={self.k}, iterations={self.iterations})'


    def silhoutte_coef(self, bool_array: np.ndarray, data_set: np.ndarray, centers: np.ndarray):
        """
        Calculates the silhoutte coef for each point in the data set
        See: https://en.wikipedia.org/wiki/Silhouette_(clustering)
        :param bool_array: The boolean array used to index the data_set to get the clusters.
        :param data_set: The data used to find the centers.
        :param centers: The centers to be moved.
        :return: None
        """
        empty_count = 0


        coef_list = []
        for i in range(int(self.k)):
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
                    for j in range(int(self.k)):
                        # loop through every outer cluster and find min distance cluster
                        if i == j:
                            continue
                        records_in_cluster_j = data_set[bool_array[j]]
                        b = sum([np.sum((r1 - r3) ** 2) ** (1/2) for r3 in records_in_cluster_j]) / len(records_in_cluster_j)

                        if b < min_b_dist:
                            min_b_dist = b

                    # return min distance to nearest cluster
                    b = min_b_dist
                    #print("a is %s -- b is %s" % (a,b))
                    sil_score = (b-a)/max(b,a)
                    coef_list.append(sil_score)
        
        mean_sil_score = sum(coef_list)/len(coef_list)
        print("mean silhoutte score is %s" % mean_sil_score)
        self.sil_coef = mean_sil_score
        return mean_sil_score

    def calc_purity(self, bool_array: np.ndarray, data_set: np.ndarray, class_labels): 
        # Adapted from https://stackoverflow.com/questions/34047540/python-clustering-purity-metric

        #print(class_labels)
        y_true = class_labels
        y_pred = []
        for c,col in enumerate(bool_array[0]):
            for r, row in enumerate(bool_array):
                if bool_array[r][c] == True:
                    y_pred.append(r)
        
        self.y_pred  = y_pred

        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
        self.purity = purity
        return purity

k_values = {
    'iris': 3,
    'synthetic': 4
}



def main(data_sets_to_process=None):
    parser = argparse.ArgumentParser(description='Dbscan')

    parser.add_argument('--max-iterations', action="store", default=2000)
    parser.add_argument('--dataset', action="store", default="iris")
    parser.add_argument('--k', action="store", default=3)


    args = parser.parse_args()
    max_iterations = int(args.max_iterations)
    #random.seed(123)
    #np.random.seed(2)
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
        k_means = KMeans(int(args.k), max_iterations)

        centers = k_means.produce(preprocessed_data_set)
        k_means_data_sets[name] = centers



    

    data_set = np.array(preprocessed_data_set)
    bool_array = k_means.center_assignment(centers,data_set)

    colors_pred = k_means.get_pred()
    colors_true = preprocessed_data_set.get_output_values()

    sil_coef = k_means.silhoutte_coef(bool_array, data_set, centers)
    purity = k_means.calc_purity(bool_array, data_set, colors_true)

    x = [row[0] for row in data_set]
    y = [row[1] for row in data_set]

    plt.scatter(x, y, c=colors_true, alpha=0.5,s=3)
    plt.suptitle('Ground-truth Data', fontsize=20)

    plt.show()

    plt.scatter(x, y, c=colors_pred, alpha=0.5,s=3)
    plt.suptitle('K-means Predicted labels', fontsize=20)
    plt.xlabel("Mean Sillhoutte: %s -- Purity: %s" % (round(k_means.sil_coef, 3), round(k_means.purity,3)))

    plt.show()

    print("centers are %s" % centers)
    print("sil_coef is %s" % sil_coef)
    print("purity is %s" % k_means.purity)

if __name__ == '__main__':
    main()
