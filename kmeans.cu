#include "kmeans.h"
#include <cublas_v2.h>

#define DIM_BLOCK 128
#define SHARED_POINTS 100

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
void nearest_cluster_new(float *point_norm, float *cluster_norm, float *pc_product,
int num_points, int num_coords, int num_clusters, float *new_clusters, 
int *membership, int *membership_changes, int *clusters_size, float *points) {

  unsigned int gdim = gridDim.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int tid = threadIdx.x;
  unsigned int num_threads = gdim * bdim;
  unsigned int obj_idx;

	extern __shared__ float shared[];
	float *s_cluster_norm = shared;

	__syncthreads();

  int s_per_time = SHARED_POINTS;
  int times = (int) num_clusters / s_per_time;

  int new_cluster_idx = 0;
  float dist, min_dist = 3.40282e+38;

  for (obj_idx = bid * bdim + tid; obj_idx < num_points; obj_idx += num_threads) {

    // save centroid norms into shared memory by tiles, and calculate distances
    for (int t = 0; t < times; t ++) {

      for (int i = tid; i < s_per_time; i += bdim) {
        s_cluster_norm[i] = cluster_norm[t * s_per_time + i];
      }

	    __syncthreads();

#ifdef DEVICE_TIMING
clock_t start;
clock_t duration;
if (tid == 0) { start = clock(); }
#endif

      for (int i = 0; i < s_per_time; i++) {
      if ((dist = point_norm[obj_idx] + s_cluster_norm[i] - 
               pc_product[i * num_points + obj_idx]) < min_dist) {
				  min_dist = dist;
				  new_cluster_idx = i;
	  		}
		  }
    }

    // process the remaining clusters.
    // The final loop is unrolled to avoid an extra comparison in the previous loops
    for (int t = times * s_per_time; t < num_clusters; t ++) {

      for (int i = tid; i < num_clusters - times * s_per_time; i += bdim) {
        s_cluster_norm[i] = cluster_norm[times * s_per_time + i];
      }

      __syncthreads();

      for (int i = 0; i < s_per_time; i++) {
      if ((dist = point_norm[obj_idx] + s_cluster_norm[i] - 
               pc_product[i * num_points + obj_idx]) < min_dist) {
				  min_dist = dist;
				  new_cluster_idx = i;
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

  	__syncthreads();

#ifdef DEVICE_TIMING
if (tid == 0) {
duration = clock() - start;
printf("\tcent cal time = %lld microseconds\n", (long long) duration);
start = clock();
}
#endif

  }
}

__global__
void nearest_cluster(float *points, float *clusters, int num_points, int num_coords, int num_clusters,
		float *new_clusters, int *membership, int *membership_changes, int *clusters_size) {

	unsigned int bid = blockIdx.x;
	unsigned int bdim = blockDim.x;
	unsigned int tid = threadIdx.x;
  unsigned int obj_idx = bid * bdim + tid;

	extern __shared__ float shared[];
	float *s_clusters = shared;

	__syncthreads();

  int s_per_time = (int) (SHARED_POINTS / (num_coords));
  int length_per_time = s_per_time * num_coords;
  int times = (int) num_clusters / s_per_time;

  int new_cluster_idx = 0;
  float dist, min_dist = 3.40282e+38;

  // save centroids into shared memory by tiles, and calculate distances
  for (int t = 0; t < times; t ++) {

    for (int i = tid; i < length_per_time; i += bdim) {
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

    for (int i = tid; i < (num_clusters - times * s_per_time) * num_coords; 
             i += bdim) {
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
	int *clusters_size;
	size_t points_length = num_points * num_coords * sizeof(float);
	size_t clusters_length = num_clusters * num_coords * sizeof(float);
  size_t pc_product_length = num_points * num_clusters * sizeof(float);
	float **retval, **clusters;
  float **trans_points;
  float **trans_clusters;
  float *pc_product;
  float *point_norm;
  float *cluster_norm;

	// transpose the points to coalesce
  trans_points = (float**) malloc(num_coords * sizeof(float*));
  assert(trans_points);
  trans_points[0] = (float*) malloc(points_length);
  assert(trans_points[0]);
  for (i = 0; i < num_coords; i++) {
    if ( i > 0) trans_points[i] = trans_points[i - 1] + num_points;
		for (j = 0; j < num_points; j++)
			trans_points[i][j] = points[j][i];
	}

	retval = (float**) malloc(num_clusters * sizeof(float*));
	assert(retval);
	retval[0] = (float*) malloc(clusters_length);
	assert(retval[0]);
	for (i = 1; i < num_clusters; i++) {
		retval[i] = retval[i - 1] + num_coords;
	}

  trans_clusters = (float**) malloc(num_coords * sizeof(float*));
  assert(trans_clusters);
  trans_clusters[0] = (float*) malloc(clusters_length);
  assert(trans_clusters[0]);

  pc_product = (float*) calloc(num_points * num_clusters, sizeof(float));

  point_norm = (float*) malloc(num_points * sizeof(float));
  cluster_norm = (float*) malloc(num_clusters * sizeof(float));

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
	clusters[0] = (float*) calloc(num_clusters * num_coords, sizeof(float));
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
  float *device_point_norm, *device_cluster_norm;
  float *d_vector;
	int *device_membership;
	int *device_membership_changes;
	int *device_clusters_size;
  float alpha = 2.0f;
  float beta = 0.0f;
  cublasStatus_t stat;
  cublasHandle_t handle;

	checkCudaError(__LINE__, cudaMalloc(&device_points, points_length));
	checkCudaError(__LINE__, cudaMalloc(&device_trans_points, points_length));
	checkCudaError(__LINE__, cudaMalloc(&device_clusters, clusters_length));
	checkCudaError(__LINE__, cudaMalloc(&device_trans_clusters, clusters_length));
	checkCudaError(__LINE__, cudaMalloc(&device_new_clusters, clusters_length));
	checkCudaError(__LINE__, cudaMalloc(&device_pc_product, pc_product_length));
  checkCudaError(__LINE__, cudaMalloc(&device_point_norm, num_points * sizeof(float)));
  checkCudaError(__LINE__, cudaMalloc(&device_cluster_norm, num_clusters * sizeof(float)));
  checkCudaError(__LINE__, cudaMalloc(&d_vector, num_coords * sizeof(float)));
	checkCudaError(__LINE__, cudaMalloc(&device_clusters_size, num_clusters * sizeof(int)));
	checkCudaError(__LINE__, cudaMalloc(&device_membership, num_points * sizeof(int)));
	checkCudaError(__LINE__, cudaMalloc(&device_membership_changes, dimGrid * sizeof(int)));

//	checkCudaError(__LINE__, cudaMemcpy(device_points, points[0],
//      points_length, cudaMemcpyHostToDevice));
	checkCudaError(__LINE__, cudaMemcpy(device_trans_points, trans_points[0],
      points_length, cudaMemcpyHostToDevice));
	checkCudaError(__LINE__, cudaMemcpy(device_membership,
			membership, num_points * sizeof(int), cudaMemcpyHostToDevice));

  stat = cublasCreate(&handle);

  for (i = 0; i < num_points; i ++) {
    stat = cublasSetVector(num_coords, sizeof(float), points[i], 1, d_vector, 1);
    stat = cublasSnrm2(handle, num_coords, d_vector, 1, &point_norm[i]);
    point_norm[i] = point_norm[i] * point_norm[i];
  }

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

//    checkCudaError(__LINE__, cudaMemset(device_pc_product, 0, 
//        pc_product_length));
		checkCudaError(__LINE__, cudaMemcpy(device_clusters, retval[0],
				clusters_length, cudaMemcpyHostToDevice));
//  	checkCudaError(__LINE__, cudaMemcpy(device_trans_clusters, trans_clusters[0], 
//        clusters_length, cudaMemcpyHostToDevice));
		checkCudaError(__LINE__, cudaMemcpy(device_new_clusters, clusters[0],
				clusters_length, cudaMemcpyHostToDevice));
		checkCudaError(__LINE__, cudaMemcpy(device_clusters_size, clusters_size,
				num_clusters * sizeof(int), cudaMemcpyHostToDevice));

#ifdef KERNAL_TIMING
int64_t duration = GetTimeMiusFrom(start);
printf("prep time = %lld microseconds\n", (long long) duration);
start = GetTimeMius64();
#endif

    memset (pc_product, 0, pc_product_length);

    // (x_i - c_j)^2 = (x_i)^2 + (c_j)^2 - 2*x_i*c_j
    // 1. Use cuBLAS to compute x_i*c_j
    stat = cublasSetMatrix(num_clusters, num_coords, sizeof(float), trans_clusters, num_clusters, device_trans_clusters, num_clusters);
    stat = cublasSetMatrix(num_coords, num_points, sizeof(float), points[0], num_coords, device_points, num_coords);
    stat = cublasSetMatrix(num_clusters, num_points, sizeof(float), pc_product, num_clusters, device_pc_product, num_clusters);

		cudaDeviceSynchronize();

    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_clusters, num_points, 
              num_coords, &alpha, device_trans_clusters, num_clusters, 
              device_points, num_coords, &beta, device_pc_product, num_clusters);

		cudaDeviceSynchronize();

    // 2. Compute (x_i)^2 and (c_j)^2
    for (i = 0; i < num_clusters; i ++) {
      stat = cublasSetVector(num_coords, sizeof(float), retval[i], 1, d_vector, 1);
      stat = cublasSnrm2(handle, num_coords, d_vector, 1, &cluster_norm[i]);
      cluster_norm[i] = cluster_norm[i] * cluster_norm[i];
    }

		cudaDeviceSynchronize();

  	checkCudaError(__LINE__, cudaMemcpy(device_point_norm, point_norm,
				num_points * sizeof(float), cudaMemcpyHostToDevice));
  	checkCudaError(__LINE__, cudaMemcpy(device_cluster_norm, cluster_norm,
				num_clusters * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Compute nearest cluster

    nearest_cluster_new
				<<<dimGrid, dimBlock, SHARED_POINTS * sizeof(float)>>>
        (device_point_norm, device_cluster_norm, device_pc_product, 
         num_points, num_coords, num_clusters, device_new_clusters, 
         device_membership, device_membership_changes, device_clusters_size, 
         device_trans_points);

#if 0
		nearest_cluster
				<<<dimGrid, dimBlock, clusters_length>>>
        (device_trans_points, device_clusters, num_points, num_coords, num_clusters,
         device_new_clusters, device_membership, device_membership_changes, 
         device_clusters_size);
#endif

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
				clusters_length, cudaMemcpyDeviceToHost));
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
	free(clusters[0]);
	free(clusters);
  free(trans_clusters[0]);
  free(trans_clusters);
  free(pc_product);
  free(point_norm);
  free(cluster_norm);
	free(clusters_size);

  checkCudaError(__LINE__, cudaFree(device_points));
	checkCudaError(__LINE__, cudaFree(device_trans_points));
	checkCudaError(__LINE__, cudaFree(device_clusters));
	checkCudaError(__LINE__, cudaFree(device_trans_clusters));
	checkCudaError(__LINE__, cudaFree(device_new_clusters));
  checkCudaError(__LINE__, cudaFree(device_pc_product));
  checkCudaError(__LINE__, cudaFree(d_vector));
  checkCudaError(__LINE__, cudaFree(device_point_norm));
  checkCudaError(__LINE__, cudaFree(device_cluster_norm));
	checkCudaError(__LINE__, cudaFree(device_membership));
	checkCudaError(__LINE__, cudaFree(device_membership_changes));
	checkCudaError(__LINE__, cudaFree(device_clusters_size));

	#ifdef SYNCOUNT
	free(tmp_membership_changes);
	#endif

  cublasDestroy(handle);

	return retval;
}
