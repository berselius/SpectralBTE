#include "mesh_setup.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

//this returns nX as a value
//nx = total number of points
//nX_node = number of points per rank
//dx_min = smallest dx
//x centerpoints of grids assigned to this rank
//dx for each centerpoint assigned to this rank
void make_mesh(int *nX, int *nX_node, double *dx_min, double **x, double **dx, int ORDER, char *meshFilename) {
  //set up spatial discretization
  //returns the minimum mesh size for checking CFL

  //NOTE - 2 possibilities
  //if first line is a nonzero positive integer, there are that many 'zones' of size L with N points, as specified on the following lines
  //if first line is zero, then every mesh point is specified in the file (currently not implemented)

  //Mesh from file
  FILE *meshFile;

  char meshFilebuffer[250];
  double x_edge = 0.0;
  int i, j, numZones, *NZone, dxCount, storeCount;
  double LZone, *dx_zone;

  int rank, numNodes;

  sprintf(meshFilebuffer, "./input/%s", meshFilename);
  printf("Opening %s\n", meshFilebuffer);

  meshFile = fopen(meshFilebuffer, "r");
  //meshFile = fopen("./input/mesh.dat","r");

  //First line is explanation
  fscanf(meshFile, "%s\n", meshFilebuffer);
  //printf("%d %s\n",test, meshfilebuffer);

  //get total number of points
  fscanf(meshFile, "%d\n", nX);

  //check number of zones
  fscanf(meshFile, "%d\n", &numZones);

  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (( (*nX) % numNodes) == 0) {
    *nX_node = (*nX) / numNodes;
  }
  else {
    printf("Please have a number of grid points evenly divisible by the number of nodes\n");
    fflush(stdout);
    exit(0);
  }

  //dx for a specific zone
  dx_zone = malloc(sizeof(double) * numZones);
  //points per zone?
  NZone = malloc(sizeof(int) * numZones);
  //printf("%d %d %d \n", rank, numZones, *nX);

  if (numZones < 1) {
    printf("%d\n", numZones);
    printf("Error - bad number of zones listed in mesh generation\n");
    exit(0);
  }

  for (i = 0; i < numZones; i++) {
    // number of points in this zone
    fscanf(meshFile, "%d\n", &NZone[i]);
    // Length of this zone
    fscanf(meshFile, "%lf\n", &LZone);
    // dx for this zone
    dx_zone[i] = LZone / (double) NZone[i];
    // this is just finding the minimum
    if ( dx_zone[i] < *dx_min)
      *dx_min = dx_zone[i];
  }


  fclose(meshFile);

  if (rank == 0) printf("allocating x, dx\n");

  int size = ((*nX_node) + (2 * ORDER));
  *x = malloc(sizeof(double) * (size)); //includes ghost cell info
  *dx = malloc(sizeof(double) * (size));
  //now start filling in the x,dx

  storeCount = 0;
  if (numNodes == 1) {
    dxCount = ORDER;
    for (i = 0; i < numZones; i++) {
      for (j = 0; j < NZone[i]; j++) {
	//only store it if we're in our current window, plus or minus one for the ghost cells
	(*dx)[dxCount] = dx_zone[i];
	(*x)[dxCount] = x_edge + 0.5 * dx_zone[i];
	//printf("%d %d %g %g\n", rank, dxCount, (*dx)[dxCount], (*x)[dxCount]);
	x_edge += dx_zone[i];
	//printf("index: %d  x:%lf\n",dxCount,x[dxCount]);
	dxCount++;
      }
    }
    if (ORDER == 1) {
      (*dx)[0] = (*dx)[1];
      (*x)[0] = (*x)[1] - (*dx)[1];
    }
    else {
      (*dx)[1] = (*dx)[2];
      (*x)[1] = (*x)[2] - (*dx)[2];
      (*dx)[0] = (*dx)[1];
      (*x)[0] = (*x)[1] - (*dx)[1];
    }
    if (ORDER == 1) {
      (*dx)[(*nX_node) + 1] = (*dx)[(*nX_node)];
      (*x)[(*nX_node) + 1] = (*x)[*nX_node] + (*dx)[(*nX_node)];
    }
    else {
      (*dx)[(*nX_node) + 2] = (*dx)[(*nX_node) + 1];
      (*x)[(*nX_node) + 2] = (*x)[(*nX_node) + 1] + (*dx)[(*nX_node) + 1];
      storeCount++;
      (*dx)[(*nX_node) + 3] = (*dx)[(*nX_node) + 2];
      (*x)[(*nX_node) + 3] = (*x)[(*nX_node) + 2] + (*dx)[(*nX_node) + 2];
    }
  }
  // standard case
  else {
    dxCount = 0;
    storeCount = 0;
    for (i = 0; i < numZones; i++) {
      for (j = 0; j < NZone[i]; j++) {
	//only store it if we're in our current window, plus or minus one for the ghost cells
	if ((dxCount >= (rank * (*nX_node) - ORDER)) && (dxCount < (rank + 1) * (*nX_node) + ORDER) ) {
	  (*dx)[storeCount] = dx_zone[i];
	  (*x)[storeCount] = x_edge + 0.5 * dx_zone[i];
	  storeCount++;
	}
	x_edge += dx_zone[i];
	//printf("index: %d  x:%lf\n",dxCount,x[dxCount]);
	dxCount++;
      }
    }

    if (ORDER == 1) {
      (*dx)[0] = (*dx)[1];
      (*x)[0] = (*x)[1] - (*dx)[1];
    }
    else {
      (*dx)[1] = (*dx)[2];
      (*x)[1] = (*x)[2] - (*dx)[2];
      (*dx)[0] = (*dx)[1];
      (*x)[0] = (*x)[1] - (*dx)[1];
    }
    if (ORDER == 1) {
      (*dx)[storeCount] = (*dx)[storeCount - 1];
      (*x)[storeCount] = (*x)[storeCount - 1] + (*dx)[storeCount - 1];
    }
    else {
      (*dx)[storeCount] = (*dx)[storeCount - 1];
      (*x)[storeCount] = (*x)[storeCount - 1] + (*dx)[storeCount - 1];
      storeCount++;
      (*dx)[storeCount] = (*dx)[storeCount - 1];
      (*x)[storeCount] = (*x)[storeCount - 1] + (*dx)[storeCount - 1];
    }
  }

  printf("Loaded mesh!\n");
  fflush(stdout);
  free(dx_zone);
  free(NZone);
}
