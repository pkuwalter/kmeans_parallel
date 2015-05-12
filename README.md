kmeans-gpu
==========================

## A baseline K-Means program on GPU

### Basic version (master branch): cuda kernel
1. Calculate the distance between each point and cluster centroids, and find the nearest centroids;
2. For each centroid, calculate the number of its belonging points, and sum up their coordinates.
3. The averaging (a final division) is done outside the kernel.



### GPU architecture-related optimization:
1. Coalesced memory access;
2. Shared memory -- save centroids into shared memory by tiles;
3. Prefetching;

  * If define SYNCOUNT in the Makefile, '\_\_syncthreads\_count' would be used to replace an atomic sumation.



### Versions and branches:
* master: Pure hand-coded version, without using CuBLAS. dist = (x_i - c_i)^2;
* v2: (x_i - c_i)^2 = x_i^2 + c_i^2 - 2x_i\*c_i, use cublasSgemm() and cublasSnrm2() to compute each tearm, finding cublasSnrm2() is slow;
* v3: Using the diagonals of (x^t \* x), computed by cublasSgemm(), to substitute cublasSnrm2(), limited by the points size;
* v4: Self-written vec_norm(), sharding vector to get smaller cublasSgemm(), solve v3's limit on size.



### Usage: 
```
./kmeans-gpu [switches] -i filename -n num\_clusters
    -i filename    : file containing data to be clustered
    -n num_clusters: number of clusters (K must > 1)
    -t threshold   : threshold value (default 0.0010)
    -c iteration   : end after iterations
```
