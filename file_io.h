/*
 * file_io.h
 *
 *  Created on: Dec 1, 2013
 *      Author: yige
 */

#ifndef FILE_IO_H_
#define FILE_IO_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cstring>

#include "kmeans.h"

using namespace std;


/*
 * read input file.
 * When allocating memory for input data, align in one-dimensional array to suit future changes to Cuda
 * Performance issue: http://en.wikipedia.org/wiki/Data_structure_alignment
 * Alternative: cudaMalloocPitch(), also has performance issue
 * @param filename input file name
 * @param num_points number of points
 * @param num_coords number of coordinates
*/
float** file_read(char *filename, int  *num_points, int  *num_coords);

/*
 * write output file.
 * @param filename input file name
 * @param num_points number of points
 * @param num_coords number of coordinates
 * @param clusters [numClusters][numCoords] clusters centers
 * @param membership [numObjs] membership of points to clusters
*/
int file_write(char *filename, int num_clusters, int num_points,
		int num_coords, float **clusters, int *membership);


#endif /* FILE_IO_H_ */
