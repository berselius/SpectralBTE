/**********

GENERATES THE WEIGHTS FOR GRAZING COLLISIONS USING OPENMP/MPI

Requires MPIcollisionRoutines object file

Command line arguments
===================
N: number of points in each velocity direction
glance: small parameter used in glancing collisions calculations. Set to 0 to explicitly compute the Landau-FP operator
beta: used for inelastic formulation: set to 1 for our purposes
lambda: exponent of relative velocity in scattering kernel
L_v: Semi-length of velocity domain [-L_v,L_v]
*************/
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include "MPIcollisionRoutines.h"
#include <mpi.h>
#include <gsl/gsl_math.h>

int N;
int GL = 64;
double glance, glancecons;

int rank, numNodes;

double *v, L_v, *eta, L_eta, h_v, h_eta;
double lambda, beta;


int main(int argc, char *argv[])
{
  //************************
  //MPI-related variables!
  //************************
  MPI_Status status;

  //------------------------------
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int i,j;
  double **conv_weights;
  double *output_buffer;

  //read in inputs
  N = atoi(argv[1]); // Knudsen Number
  glance = (double)atof(argv[2]); //glancing parameter
  beta = (double)atof(argv[3]); // 1 = elastic; 0.5 = sticky inelastic
  lambda = (double)atof(argv[4]); // = 0: MM; = 1: HS  (\lambda variable hard potentials)
  L_v = (double)atof(argv[5]); // Size of domain

  //check to make sure we have a legit MPI setup
  if((N*N*N % numNodes) != 0) {
    printf("Error: numNodes must divide N^3\n");
    exit(0);
  }

  v = malloc(N*sizeof(double));
  eta = malloc(N*sizeof(double));
  output_buffer = malloc(N*N*N*sizeof(double));

  h_v = 2.0*L_v/(double)(N-1);
  h_eta = (2.0*M_PI/N) / h_v;
  L_eta = 0.5*N*h_eta;

    
  for(i=0;i<N;i++)    {
    eta[i] = -L_eta + (double)i*h_eta;
    v[i] = -L_v + (double)i*h_v;
  }

  //convolution weight info
  char buffer_weights[100];
  if(glance == 0)
    sprintf(buffer_weights,"Weights/N%d_Aniso_L_v%g_lambda%g_Landau.wts",N,L_v,lambda);
  else
    sprintf(buffer_weights,"Weights/N%d_AnIso_L_v%g_lambda%g_glance%g.wts",N,L_v,lambda,glance);

  //generate convolution weights - only do an even chunk of them
  conv_weights = malloc((N*N*N/numNodes)*sizeof(double *));
  for (i=0;i<(N*N*N/numNodes);i++) conv_weights[i] = malloc(N*N*N*sizeof(double));
 
  FILE *fidWeights; 
    //check to see if these convolution weights have already been pre-computed
  if( (fidWeights = fopen(buffer_weights,"r")) ) { //checks if we've already stored a file
    printf("Stored weights already computed! \n");
    fclose(fidWeights);
  }      
  else {
    if(rank == 0) 
      printf("Stored weights not found, generating...\n");
 
    generate_conv_weights(conv_weights); 
    
    MPI_Barrier(MPI_COMM_WORLD);
    //get weights from everyone else...

    if(rank == 0) {
      //dump the weights we've computed into a file
      fidWeights = fopen(buffer_weights,"w");
      for(i=0;i<(N*N*N/numNodes);i++) {
	fwrite(conv_weights[i],sizeof(double),N*N*N,fidWeights);
      } 
      //receive from all other processes
      for(i=1;i<numNodes;i++) {
	for(j=0;j<(N*N*N/numNodes);j++) {
	  MPI_Recv(output_buffer,N*N*N,MPI_DOUBLE,i,j + i*N*N*N/numNodes,MPI_COMM_WORLD,&status);
	  fwrite(output_buffer,sizeof(double),N*N*N,fidWeights);
	}
      }
      if(fflush(fidWeights) != 0) {
	printf("Something is wrong with storing the weights");
	exit(0);
      } 
      fclose(fidWeights);
    }
    else {
      for(i=0;i<N*N*N/numNodes;i++)
	MPI_Send(conv_weights[i],N*N*N,MPI_DOUBLE,0,rank*(N*N*N/numNodes)+i,MPI_COMM_WORLD);
    }
  }
    
  MPI_Finalize();

  return 0;
}
