#include "kmeans.h"
#include <cublas_v2.h>

#define DIM_BLOCK 128

__host__ __device__
inline float dist_square(int dimension, int num_points, float *points, int obj_idx, float *p2) {
	float ans = 0.0, tmp;
	float coord, coord_next;

	// prefetch
	coord_next = points[obj_idx];

	for (int i = 0; i < dimension; i++) {
		coord = coord_next;
		coord_next = points[(i + 1) * num_points + obj_idx];  // transposed
		tmp = coord - p2[i];
		ans += tmp * tmp;
	}
	return ans;
}

__global__
void nearest_cluster(float *points, float *clusters, int num_points, int num_coords, int num_clusters,
		float *new_clusters, int *membership, int *membership_changes, int *clusters_size) {

	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int tid = threadIdx.x;
	//unsigned int clusters_length = num_clusters * num_coords;
  unsigned int obj_idx = bid * bdim + tid;

	extern __shared__ float shared[];
	float *s_clusters = shared;

	__syncthreads();

  int s_per_time = (int) (100 / (num_coords));
  int length_per_time = s_per_time * num_coords;
  int times = (int) num_clusters / s_per_time;

  int new_cluster_idx = 0;
  float dist, min_dist = 3.40282e+38;

  // save centroids into shared memory by tiles, and calculate distances
  for (int t = 0; t < times; t ++) {

    for (int i = tid; i < length_per_time; i ++) {
      s_clusters[i] = clusters[t * s_per_time + i];
    }

	  __syncthreads();

#ifdef DEVICE_TIMING
clock_t start;
clock_t duration;
if (tid == 0) { start = clock(); }
#endif

	  if (obj_idx < num_points) {

      for (int i = 0; i < s_per_time; i++) {
      if ((dist = dist_square(num_coords, num_points, points, obj_idx, 
            &s_clusters[i * num_coords]))
			  		< min_dist) {
				  min_dist = dist;
				  new_cluster_idx = i;
	  		}
		  }
    }
  }

  // process the remaining clusters.
  // The final loop is unrolled to avoid an extra comparison in the previous loops
  for (int t = times * s_per_time; t < num_clusters; t ++) {

    for (int i = tid; i < (num_clusters - times * s_per_time) * num_coords; i ++) {
      s_clusters[i] = clusters[times * s_per_time + i];
    }

    __syncthreads();

    if (obj_idx < num_points) {

      for (int i = 0; i < num_clusters - times * s_per_time; i++) {
      if ((dist = dist_square(num_coords, num_points, points, obj_idx, 
            &s_clusters[i * num_coords]))
            < min_dist) {
          min_dist = dist;
          new_cluster_idx = i;
        }
      }
    }

  }

#ifdef DEVICE_TIMING
if (tid == 0) {
duration = clock() - start;
printf("\tdist time = %lld microseconds\n", (long long) duration);
start = clock();
}
#endif

  if (obj_idx < num_points) {
    int old_cluster_idx = membership[obj_idx];
	  #ifdef SYNCOUNT
  	membership_changes[bid] = __syncthreads_count(old_cluster_idx != new_cluster_idx);
  	#endif
	  if (old_cluster_idx != new_cluster_idx) {
		  #ifndef SYNCOUNT
    	atomicAdd(membership_changes, 1);
	  	#endif
		  membership[obj_idx] = new_cluster_idx;
    }

#ifdef DEVICE_TIMING
if (tid == 0) {
duration = clock() - start;
printf("\tmemb cal time = %lld microseconds\n", (long long) duration);
start = clock();
}
#endif

  	atomicAdd(&clusters_size[new_cluster_idx], 1);
	  for (int i = 0; i < num_coords; i++) {
		  atomicAdd(&new_clusters[new_cluster_idx * num_coords + i], 
            points[i * num_points + obj_idx]);
  	}
  }

	__syncthreads();

#ifdef DEVICE_TIMING
if (tid == 0) {
duration = clock() - start;
printf("\tcent cal time = %lld microseconds\n", (long long) duration);
start = clock();
}
#endif

}

inline void checkCudaError(cudaError_t error) {
	if (error != cudaSuccess)
	{
		printf("cuda error code %d: %s\n", error, cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

inline void checkCudaError( int line, cudaError_t error) {
  if (error != cudaSuccess)
  {
    printf("cuda error code %d, line(%d): %s\n", error, line, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

float **kmeans(float **points, int num_points, int num_coords, int num_clusters,
			float threshold, int iterations, int *membership) {

	// initialization
	int i, j;
	float **retval, **clusters;
	int *clusters_size;
	unsigned int points_length = num_points * num_coords;
	unsigned int clusters_length = num_clusters * num_coords;

	// transpose the points to coalesce
  float **trans_points = (float**) malloc(num_coords * sizeof(float*));
  assert(trans_points);
  trans_points[0] = (float*) malloc(points_length * sizeof(float));
  assert(trans_points[0]);
  for (i = 0; i < num_coords; i++) {
    if ( i > 0) trans_points[i] = trans_points[i - 1] + num_points;
		for (j = 0; j < num_points; j++)
			trans_points[i][j] = points[j][i];
	}

	retval = (float**) malloc(num_clusters * sizeof(float*));
	assert(retval);
	retval[0] = (float*) malloc(clusters_length * sizeof(float));
	assert(retval[0]);
	for (i = 1; i < num_clusters; i++) {
		retval[i] = retval[i - 1] + num_coords;
	}

  float **trans_clusters = (float**) malloc(num_coords * sizeof(float*));
  assert(trans_clusters);
  trans_clusters[0] = (float*) malloc(clusters_length * sizeof(float));
  assert(trans_clusters[0]);

	// randomly choose initial clusters
	for (i=0; i < num_clusters; i++) {
    		memcpy(retval[i], points[i], num_coords * sizeof(float));
	}

	memset (membership, -1, sizeof(membership));

	// allocate space for temp clusters
	clusters_size = (int*) calloc(num_clusters, sizeof(int));
	assert(clusters_size);

	clusters = (float**) malloc(num_clusters * sizeof(float*));
	assert(clusters);
	clusters[0] = (float*) calloc(clusters_length, sizeof(float));
	assert(clusters[0]);
	for (i = 1; i < num_clusters; i++) {
		clusters[i] = clusters[i - 1] + num_coords;
	}

  // Prepare computation kernel

	const unsigned int dimBlock = DIM_BLOCK;
	const unsigned int dimGrid = (num_points - 1) / dimBlock + 1;

	#ifdef SYNCOUNT
	int *tmp_membership_changes = (int*) calloc(dimGrid, sizeof(int));
	assert(tmp_membership_changes);
	#endif

	// Cuda device memory allocation
	float *device_points, *device_trans_points;
	float *device_clusters, *device_trans_clusters;
	float *device_new_clusters;
  float *device_pc_product;
	int *device_membership;
	int *device_membership_changes;
	int *device_clusters_size;

  int pc_product_size = num_points * num_clusters;
  float alpha = 2.0f;
  float beta = 1.0f;
  cublasStatus_t stat;
  cublasHandle_t handle;

	checkCudaError(__LINE__, cudaMalloc(&device_points, points_length * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_trans_points, points_length * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_clusters, clusters_length * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_trans_clusters, clusters_length * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_new_clusters, clusters_length * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_pc_product, pc_product_size * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_clusters_size, num_clusters * sizeof(int)));
	checkCudaError(__LINE__, cudaMalloc(&device_membership, num_points * sizeof(int)));
	checkCudaError(__LINE__, cudaMalloc(&device_membership_changes, dimGrid * sizeof(int)));

	checkCudaError(__LINE__, cudaMemcpy(device_points, points[0],
      points_length * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaError(__LINE__, cudaMemcpy(device_trans_points, trans_points[0],
      points_length * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaError(__LINE__, cudaMemcpy(device_membership,
			membership, num_points * sizeof(int), cudaMemcpyHostToDevice));

	// K-mean calculation
	int iter = 0;
	int membership_changes = num_points;
	
	while (((float) membership_changes / (float) num_points > threshold) && (iter++ < iterations)) {

		DEBUG_LOG("iteration=%d, threshold=%5.5f\n", iter, (float) membership_changes / (float) num_points);
		membership_changes = 0;

#ifdef KERNAL_TIMING
int64_t start = GetTimeMius64();
#endif


		// initialize
		#ifndef SYNCOUNT
    checkCudaError(__LINE__, cudaMemcpy(device_membership_changes, &membership_changes,
        sizeof(int), cudaMemcpyHostToDevice));
		#endif

  	// transpose the clusters to coalesce
    for (i = 0; i < num_coords; i++) {
      if ( i > 0) trans_clusters[i] = trans_clusters[i - 1] + num_clusters;
		  for (j = 0; j < num_clusters; j++)
			  trans_clusters[i][j] = retval[j][i];
	  }

    checkCudaError(__LINE__, cudaMemset(device_pc_product, 0, 
        pc_product_size * sizeof(float)));
		checkCudaError(__LINE__, cudaMemcpy(device_clusters, retval[0],
				clusters_length * sizeof(float), cudaMemcpyHostToDevice));
  	checkCudaError(__LINE__, cudaMemcpy(device_trans_clusters, trans_clusters[0], 
        clusters_length * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaError(__LINE__, cudaMemcpy(device_new_clusters, clusters[0],
				clusters_length * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaError(__LINE__, cudaMemcpy(device_clusters_size, clusters_size,
				num_clusters * sizeof(int), cudaMemcpyHostToDevice));

#ifdef KERNAL_TIMING
int64_t duration = GetTimeMiusFrom(start);
printf("prep time = %lld microseconds\n", (long long) duration);
start = GetTimeMius64();
#endif


    // (x_i - c_j)^2 = (x_i)^2 + (c_j)^2 - 2*x_i*c_j
    // First use cuBLAS to compute x_i*c_j

#if 0
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_clusters, num_points, 
              num_coords, &alpha, device_trans_clusters, num_clusters, 
              device_points, num_coords, &beta, device_pc_product, num_clusters);
#endif

//   stat = cublasGetMatrix (m,n, sizeof (*c) ,d_c ,m,c,m); // cp d_c - >c



		// call kernel function
		nearest_cluster
				<<<dimGrid, dimBlock, clusters_length * sizeof(float)>>>
        (device_trans_points, device_clusters, num_points, num_coords, num_clusters, device_new_clusters,
        device_membership, device_membership_changes, device_clusters_size);

		cudaDeviceSynchronize();
		checkCudaError(__LINE__, cudaGetLastError());

#ifdef KERNAL_TIMING
duration = GetTimeMiusFrom(start);
printf("kernal time = %lld microseconds\n", (long long) duration);
start = GetTimeMius64();
#endif

		checkCudaError(__LINE__, cudaMemcpy(clusters_size, device_clusters_size,
        num_clusters * sizeof(int), cudaMemcpyDeviceToHost));
		checkCudaError(__LINE__, cudaMemcpy(clusters[0], device_new_clusters,
				clusters_length * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaError(__LINE__, cudaMemcpy(membership, device_membership,
				num_points * sizeof(int), cudaMemcpyDeviceToHost));
		#ifdef SYNCOUNT
		checkCudaError(__LINE__, cudaMemcpy(tmp_membership_changes, device_membership_changes,
        dimGrid * sizeof(int), cudaMemcpyDeviceToHost));

		for (i = 0; i < dimGrid; i ++) {
			membership_changes += tmp_membership_changes[i];
		}
		#else
    checkCudaError(__LINE__, cudaMemcpy(&membership_changes, device_membership_changes,
        sizeof(int), cudaMemcpyDeviceToHost));
		#endif

#ifdef KERNAL_TIMING
duration = GetTimeMiusFrom(start);
printf("end cpy time = %lld microseconds\n", (long long) duration);
start = GetTimeMius64();
#endif

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

#ifdef KERNAL_TIMING
duration = GetTimeMiusFrom(start);
printf("centroid cal time = %lld microseconds\n", (long long) duration);
#endif

	}

	free(trans_points[0]);
  free(trans_points);
  free(trans_clusters[0]);
  free(trans_clusters);

  checkCudaError(__LINE__, cudaFree(device_points));
	checkCudaError(__LINE__, cudaFree(device_trans_points));
	checkCudaError(__LINE__, cudaFree(device_clusters));
	checkCudaError(__LINE__, cudaFree(device_trans_clusters));
	checkCudaError(__LINE__, cudaFree(device_new_clusters));
  checkCudaError(__LINE__, cudaFree(device_pc_product));
	checkCudaError(__LINE__, cudaFree(device_membership));
	checkCudaError(__LINE__, cudaFree(device_membership_changes));
	checkCudaError(__LINE__, cudaFree(device_clusters_size));

	free(clusters[0]);
	free(clusters);
	free(clusters_size);

	#ifdef SYNCOUNT
	free(tmp_membership_changes);
	#endif

	return retval;
}
