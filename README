kmeans-gpu
==========================

A baseline K-Means program on GPU

The cuda kernel is invoked to:
1. Calculate the distance between each point and cluster centroids, and find the nearest centroids;
2. For each centroid, calculate the number of its belonging points, and sum up their coordinates.
3. The averaging (a final division) is done outside the kernel.



Techniques used for speeding-up:
1. Coalesced memory access;
2. Shared memory -- save centroids into shared memory by tiles;
3. Prefetching;

If define SYNCOUNT in the Makefile, '__syncthreads_count' would be used to replace an atomic sumation.



Usage: ./kmeans-gpu [switches] -i filename -n num_clusters
       -i filename    : file containing data to be clustered
       -n num_clusters: number of clusters (K must > 1)
       -t threshold   : threshold value (default 0.0010)
       -c iteration   : end after iterations
