#include <mpi.h>

void getBounds (int* lower, int* upper, int N, MPI_Comm* worker);

void resetQ(double*** Q, int num_species, int N);

void fcopy(double* buffer, double*** f, int x, int y, int z, int direction);

void qcopy(double* buffer, double*** q, int x, int y, int direction);
