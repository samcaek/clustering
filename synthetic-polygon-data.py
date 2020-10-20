import inspect
import os
import logging
import math
import random
import sys
from time import time
import argparse

import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt





def main():
    parser = argparse.ArgumentParser(description='Synthetic dataset generator')

    parser.add_argument('--num-points', action="store", required=True)
    parser.add_argument('--num-polygons', action="store", default = 4)
    parser.add_argument('--noise-proportion', action="store", default = 0.2)
    parser.add_argument('--polygon-vert-num', action="store", default=5)

    random.seed(1)
    np.random.seed(1)
    args = parser.parse_args()
    num_points = int(args.num_points)
    num_polygons = int(args.num_polygons)
    noise_proportion = float(args.noise_proportion)
    synthetic_data = np.zeros((int(num_points), 3), dtype='bool')

    create_polygon(10)
    #write_file(bool_array)
    polygons = create_non_overlapping_poly(num_polygons, int(args.polygon_vert_num))
    print(polygons)

    point_list = []
    while len(point_list) <= num_points:
        random_point = Point(random.uniform(0, 1),random.uniform(0, 1))
        for i,polygon in enumerate(polygons):
            if random_point.within(polygon):
                point_list.append([random_point.x, random_point.y,i])
    
    if noise_proportion > 0:
        noise_points = []
        while len(noise_points) < num_points * noise_proportion:
            random_point = Point(random.uniform(0, 1),random.uniform(0, 1))
            in_polygon = False
            for i,polygon in enumerate(polygons):
                if random_point.within(polygon):
                    in_polygon = True
            if not in_polygon:
                point_list.append([random_point.x, random_point.y, -1])
                noise_points.append([random_point.x, random_point.y, -1])

    x = [row[0] for row in point_list]
    y = [row[1] for row in point_list]
    colors = [row[2] for row in point_list]

    plt.scatter(x, y, c=colors, alpha=0.5,s=3)
    plt.show()

    write_file(point_list, "")



def create_non_overlapping_poly(num_polygons, vertex_num):
    # n : num of polygons
    print("Attempting to create %s non-overlapping polygons" % num_polygons)
    polygon_list = []
    n = 0
    i = 0
    while n < num_polygons:
        i += 1
        print("attempt %s" % i)
        new_poly = create_polygon(vertex_num)
        overlap = False
        for old_poly in polygon_list:
            if polygon_overlap(old_poly, new_poly):
                overlap = True
        if not overlap:
            polygon_list.append(new_poly)
            n += 1

    return polygon_list

def polygon_overlap(poly_1, poly_2):
    return poly_1.intersects(poly_2)

def create_polygon(n_dim):
    # https://automating-gis-processes.github.io/CSC18/lessons/L4/point-in-polygon.html
    coords = []
    for point in range(n_dim):
        x_coord = random.uniform(0, 1)
        y_coord = random.uniform(0, 1)
        coords.append((x_coord,y_coord))
    return Polygon(coords)

def write_file(list_of_lists, filename):
    """
    Writes the contained DataSet to a file.
    :param filename: Name of file to write to.
    """
    frame = inspect.stack()[1]
    this_dir = os.path.dirname(os.path.abspath(frame[0].f_code.co_filename))
    filename = this_dir + "/data/original/synthetic_data/synthetic.data"
    logging.info(f"\nWriting DataSet to file '{filename}'")
    with open(filename, 'w') as f:
        for row in list_of_lists:
            line = str(row[0]) + ',' + str(row[1]) + ',' + str(row[2]) + '\n'
            f.write(line)






if __name__ == "__main__":
    main()



