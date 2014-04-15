#ifndef KMEANS_H_
#define KMEANS_H_

#ifdef DEBUG
#define DEBUG_LOG(string, ...) printf(string, __VA_ARGS__)
#else
#define DEBUG_LOG(string, ...)
#endif

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <assert.h>
#include "mtime.h"

using namespace std;

/*
 * calculation for K-means.
 * Using Lloyd's algorithm: http://en.wikipedia.org/wiki/Lloyd%27s_algorithm
 * Parallel reduction using shared memory:
 * http://www.uni-graz.at/~haasegu/Lectures/GPU_CUDA/Lit/reduction.pdf?page=35
 * @param points input points
 * @param num_points number of points
 * @param num_coords number of coordinates
 * @param num_clusters number of clusters
 * @param threshold threshold to stop calculation
 * @param membership output pointer to membership results
 * @param iterations number of iterations
 */
float **kmeans(float **points, int num_points, int num_coords, int num_clusters,
			float threshold, int iterations, int *membership);

#endif /* KMEANS_H_ */
