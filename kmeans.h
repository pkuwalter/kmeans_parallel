/*
 * kmeans.h
 *
 *  Created on: Nov 30, 2013
 *      Author: yige
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#ifdef DEBUG
#define DEBUG_LOG(string, ...) printf(string, __VA_ARGS__)
#else
#define DEBUG_LOG(string, ...)
#endif

#include <cstdlib>
#include <cstring>
#include <limits>
#include <assert.h>

using namespace std;

/*
 * calculation for Euclidean distance between two points
 * @param dimension dimension of point coordinates
 */
inline float dist_square(int dimension, float *p1, float *p2);

/*
 * get the nearest cluster index of a given point, from the clusters
 * @param num_clusters number of clusters
 * @param point the given point
 * @param clusters previous cluster centers
 */
inline int nearest_cluster(int num_clusters, int num_coords, float *point, float **clusters);

/*
 * calculation for K-means.
 * Using Lloyd's algorithm: http://en.wikipedia.org/wiki/Lloyd%27s_algorithm
 * @param points input points
 * @param num_points number of points
 * @param num_coords number of coordinates
 * @param num_clusters number of clusters
 * @param threshold threshold to stop calculation
 * @param membership output pointer to membership results
 * @param iterations number of iterations
 */
float **kmeans(float **points, int num_points, int num_coords, int num_clusters,
			float threshold, int *membership, int *iterations);

#endif /* KMEANS_H_ */
