#include "kmeans.h"

__host__ __device__
inline float dist_square(int dimension, float *p1, float *p2) {
	float ans = 0.0;
	int i;
	for (i = 0; i < dimension; i++) {
		ans += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	}
	return ans;
}

__global__
void nearest_cluster(float *points, float *clusters, int num_points, int num_coords,
		int num_clusters, int *membership, int *membership_changes, int *clusters_size) {

	int obj_idx = blockIdx.x * blockDim.x + threadIdx.x;

	int new_idx = 0;
	float dist, min_dist = 3.40282e+38;

	int i;
	for (i = 0; i < num_clusters; i++) {
		if ((dist = dist_square(num_coords, &points[obj_idx], &clusters[i])) < min_dist) {
			min_dist = dist;
			new_idx = i;
		}
	}

	if (membership[obj_idx] != new_idx) {
		membership_changes++;
		membership[obj_idx] = new_idx;
	}

	clusters_size[new_idx]++;
	for (i = 0; i < num_coords; i++) {
		clusters[new_idx + i] += points[obj_idx + i];
	}
}

float **kmeans(float **points, int num_points, int num_coords, int num_clusters,
			float threshold, int *membership, int *iterations) {

	// initialization
	int i, j;
	float **retval;
	retval = (float**) malloc(num_clusters * sizeof(float*));
	assert(retval);
	retval[0] = (float*) malloc(num_clusters * num_coords * sizeof(float));
	assert(retval[0]);
	for (i = 1; i < num_clusters; i++) {
		retval[i] = retval[i - 1] + num_coords;
	}

	memset (membership, -1, sizeof(membership));

	// randomly choose initial clusters
    for (i=0; i < num_clusters; i++) {
    	memcpy(retval[i], points[i], num_coords * sizeof(float));
    }

    // allocate space for temp clusters
    int *clusters_size = (int*) calloc(num_clusters, sizeof(int));
    assert(clusters_size);

    float  **clusters = (float**) malloc(num_clusters * sizeof(float*));
	assert(clusters);
	clusters[0] = (float*) calloc(num_clusters * num_coords, sizeof(float));
	assert(clusters[0]);
	for (i = 1; i < num_clusters; i++) {
		clusters[i] = clusters[i - 1] + num_coords;
	}

	// K-mean calculation
	int iter = 1;
	int membership_changes = num_points;
	while (((float) membership_changes / (float) num_points > threshold) && (iter++ < 500)) {
		membership_changes = 0;

		// re-allocate cluster membership
		for (i = 0; i < num_points; i++) {
			int cl_idx = nearest_cluster(num_clusters, num_coords, points[i], retval);


		}

		// calculate new cluster centers
		for (i = 0; i < num_clusters; i++) {
			for (j = 0; j < num_coords; j++) {
				if (clusters_size[i] > 0) {
					retval[i][j] = clusters[i][j] / clusters_size[i];
				}
				clusters[i][j] = 0.0;
			}
			clusters_size[i] = 0;
		}
	}

	*iterations = iter;

	free(clusters[0]);
	free(clusters);
	free(clusters_size);

	return retval;
}
