#include "file_io.h";

float** file_read(char *filename, int  *num_points, int  *num_coords) {
	float** retval;
	*num_points = 0;
	*num_coords = 0;

	ifstream infile(filename);
	if (! infile.good()) {
		cout<<filename<<" not exists."<<endl;
		exit(-1);
	}
	string line;

	// get the number of coordinates
	(*num_coords) = 0;
	getline(infile, line);
	char *dup = strdup(line.c_str());
	if (strtok(dup, " \t\n") != 0) {
		// ignore the point index
		while (strtok(NULL, " ,\t\n") != NULL)
			(*num_coords)++;
	}
	free(dup);

	// get the number of points
	*num_points = 1;
	while (getline(infile, line)) {
		(*num_points)++;
	}

	DEBUG_LOG("File %s numObjs   = %d\n", filename, *num_points);
	DEBUG_LOG("File %s numCoords = %d\n", filename, *num_coords);

	// memory allocation
	int length = (*num_points) * (*num_coords);
	retval = (float**) malloc((*num_points) * sizeof(float*));
	assert(retval);
	retval[0] = (float*) malloc(length * sizeof(float));
	assert(retval[0]);
	for (int i = 1; i < (*num_points); i++)
		retval[i] = retval[i - 1] + (*num_coords);

	// rewind and read all the points
	infile.clear();
	infile.seekg(0);

	int i = 0, j;

	while (getline(infile, line)) {
		dup = strdup(line.c_str());
		if (strtok(dup, " \t\n") == NULL)
			continue;
		for (j = 0; j < (*num_coords); j++) {
			retval[i][j] = atof(strtok(NULL, " ,\t\n"));
		}
		i++;
	}
	free(dup);

	infile.close();
	return retval;
}

int file_write(char *filename, int num_clusters, int num_points,
		int num_coords, float **clusters, int *membership) {

    FILE *outfile;
    char outFileName[1024];
    int i, j;

    // output the coordinates of the cluster centres
    sprintf(outFileName, "%s-cluster_centres.out", filename);
    DEBUG_LOG("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
           num_clusters, outFileName);
    outfile = fopen(outFileName, "w");
    for (i=0; i<num_clusters; i++) {
        fprintf(outfile, "%d ", i);
        for (j=0; j<num_coords; j++) {
            fprintf(outfile, "%f ", clusters[i][j]);
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);

    // output the membership of each points
    sprintf(outFileName, "%s-membership.out", filename);
    DEBUG_LOG("Writing membership of N=%d data objects to file \"%s\"\n",
    		num_points, outFileName);
    outfile = fopen(outFileName, "w");
    for (i=0; i<num_points; i++)
        fprintf(outfile, "%d %d\n", i, membership[i]);
    fclose(outfile);

    return 1;

}

