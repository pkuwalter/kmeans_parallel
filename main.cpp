#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "file_io.h"
#include "kmeans.h"

static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename -n num_clusters\n"
        "       -i filename    : file containing data to be clustered\n"
        "       -n num_clusters: number of clusters (K must > 1)\n"
        "       -t threshold   : threshold value (default %.4f)\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}

int main(int argc, char **argv) {
	// read flags
    int c;
	float threshold = 0.001;
    int num_clusters = 0;
    char *infile = NULL;

    while ( (c=getopt(argc,argv,"i:n:t:"))!= EOF) {
        switch (c) {
            case 'i': infile=optarg;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': num_clusters = atoi(optarg);
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }

	if (infile == 0 || num_clusters <= 1)
		usage(argv[0], threshold);

	// read data from input file
	int num_points, num_coords;
	float **points = file_read(infile, &num_points, &num_coords);
	assert(points);

	// K-means calculation
	int *membership = (int*) malloc(num_points * sizeof(int));
	assert(membership);
	int iterations;
	float **clusters = kmeans(points, num_points, num_coords, num_clusters,
			threshold, membership, &iterations);

	free (points[0]);
	free(points);

	// write results to output file
	file_write(infile, num_clusters, num_points, num_coords, clusters, membership);

	free (membership);
	free (clusters[0]);
	free(clusters);
}
