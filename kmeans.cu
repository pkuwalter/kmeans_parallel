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
void nearest_cluster(float *points, float *clusters, int num_points, int num_coords,int num_clusters,
		float *new_clusters, int *membership, int *membership_changes, int *clusters_size) {

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

inline void checkCudaError(cudaError_t error) {
	if (error != cudaSuccess)
	{
		printf("cuda error code %d, line(%d): %s\n", error, __LINE__, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
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

	// randomly choose initial clusters
    for (i=0; i < num_clusters; i++) {
    	memcpy(retval[i], points[i], num_coords * sizeof(float));
    }

	memset (membership, -1, sizeof(membership));

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

	// Cuda device memory allocation

    float *device_points;
	float *device_clusters;
	float *device_new_clusters;
	int *device_membership;
	int *device_membership_changes;
	int *device_clusters_size;

	checkCudaError(cudaMalloc(&device_points, num_points * num_coords * sizeof(float)));
	checkCudaError(cudaMalloc(&device_clusters, num_clusters * num_coords * sizeof(float)));
	checkCudaError(cudaMalloc(&device_new_clusters, num_clusters * num_coords * sizeof(float)));
	checkCudaError(cudaMalloc(&device_membership, num_points * sizeof(int)));
	checkCudaError(cudaMalloc(&device_membership_changes, sizeof(int)));
	checkCudaError(cudaMalloc(&device_clusters_size, num_clusters * sizeof(int)));

	checkCudaError(cudaMemcpy(device_points, points[0],num_points * num_coords * sizeof(float),	cudaMemcpyHostToDevice));
	checkCudaError(cudaMemcpy(device_membership, membership, num_points * sizeof(int), cudaMemcpyHostToDevice));

	const unsigned int dimBlock = 128;
	const unsigned int dimGrid = (num_points - 1) / dimBlock + 1;

	// K-mean calculation
	int iter = 1;
	int membership_changes = num_points;
	while (((float) membership_changes / (float) num_points > threshold) && (iter++ < 500)) {
		membership_changes = 0;

		// initialize
		checkCudaError(cudaMemcpy(device_clusters, retval[0], num_clusters * num_coords * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaError(cudaMemcpy(device_new_clusters, clusters[0], num_clusters * num_coords * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaError(cudaMemcpy(device_clusters_size, clusters_size, num_clusters * sizeof(int), cudaMemcpyHostToDevice));

		// call kernel function
		nearest_cluster<<<dimGrid, dimBlock>>>
		(device_points, device_clusters, num_points, num_coords, num_clusters, device_new_clusters,
				device_membership, device_membership_changes, device_clusters_size);

		cudaDeviceSynchronize();
		checkCudaError(cudaGetLastError());

		checkCudaError(cudaMemcpy(clusters[0], device_new_clusters, num_clusters * num_coords * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(membership, device_membership, num_points * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(&membership_changes, device_membership_changes, sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaError(cudaMemcpy(clusters_size, device_clusters_size, num_clusters * sizeof(int), cudaMemcpyDeviceToHost));


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
