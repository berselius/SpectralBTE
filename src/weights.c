#include "weights.h"
#include <math.h>
#include "gauss_legendre.h"
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include "species.h"
#include <string.h>
#include <mpi.h>

static int N;
static double L_v;
static double *zeta;
static double deta;
static double prefactor;
static double lambda;
static double diam_i;
static double diam_j;
static double mass_i;
static double mass_j;
static double mu_ij;
//static gsl_integration_glfixed_table *GL_table;
//static double max = 0.0;
static double *wtN;

//Internal functions
static int load(double **conv_weights, char buffer_weights[100], int N, int lower, int range);

void write_weights(double **conv_weights, char buffer_weights[100], int N, int lower, int range, int size, int rank);

void write_zeroD(double **conv_weights, char buffer_weights[100], int N);

static void generate_conv_weights_iso(int lower, int range, double **conv_weights);

static double ghat(double r, void *args);

static double gHat3(double ki1, double ki2, double ki3, double zeta1, double zeta2, double zeta3);

// Allocates arrays N*N*N*total_species
void alloc_weights(int range, double ****conv_weights, int total_species) {
  int i;
  *conv_weights = malloc(total_species * sizeof(double **));
  for (i = 0; i < total_species; i++) {
    (*conv_weights)[i] = malloc(range * sizeof(double *));
  }
}

// weightFlag = recompute weights even if they are already saved
void initialize_weights(int lower, int range, int nodes, double *eta, double Lv, double lam, int weightFlag, int isoFlag, double **conv_weights, species species_i, species species_j, int size, int rank) {
  char buffer_weights[100];
  int i;

  N = nodes;
  zeta = eta;
  deta = eta[1] - eta[0];
  L_v = Lv;
  lambda = lam;
  prefactor = 16.0 * M_PI * M_PI * deta * deta * deta / pow(2.0 * M_PI, 1.5) / (4.0 * M_PI);
  diam_i = species_i.d_ref;
  diam_j = species_j.d_ref;
  mass_i = species_i.mass;
  mass_j = species_j.mass;
  mu_ij = mass_i * mass_j / (mass_i + mass_j);

  wtN = malloc(N * sizeof(double));
  wtN[0] = 0.5;
  for (i = 1; i < (N - 1); i++) {
    wtN[i] = 1.0;
  }
  wtN[N - 1] = 0.5;

  //GL_table = gsl_integration_glfixed_table_alloc(64);

  printf("%g %g %g %g %s %s \n", diam_i, diam_j, mass_i, mass_j, species_i.name, species_j.name);

  for (i = 0; i < range; i++) {
    conv_weights[i] = malloc(N * N * N * sizeof(double));
  }

  if (!isoFlag) {
    if (strcmp(species_i.name, "default") == 0) {
      sprintf(buffer_weights, "Weights/N%d_isotropic_L_v%g_lambda%g.wts", N, L_v, lambda); //old style of naming
    }
    else {
      sprintf(buffer_weights, "Weights/N%d_isotropic_L_v%g_HS_%s_%zd_%s_%zd.wts", N, L_v, species_i.name, species_i.id, species_j.name, species_j.id);
    }

    if(weightFlag == 0) {
      if(!load(conv_weights, buffer_weights, N,lower,range)) {
	printf("Stored weights not found for this configuration, generating ...\n");
	generate_conv_weights_iso(lower, range, conv_weights);
	write_weights(conv_weights, buffer_weights, N,lower,range,size,rank);
      }
    }
    else {//weights forced to be regenerated
      printf("Fresh version of weights being computed and stored for this configuration\n");
      generate_conv_weights_iso(lower, range, conv_weights);
      //dump the weights we've computed into a file
      write_weights(conv_weights, buffer_weights, N,lower,range,size,rank);
    }
  }
  else {
    sprintf(buffer_weights, "Weights/N%d_AnIso_L_v%g_lambda%g_Landau.wts", N, L_v, lambda);
    //sprintf(buffer_weights,"Weights/N%d_AnIso_L_v%g_lambda%g_glance0.0001_C.wts",N, L_v,lambda);
    if(weightFlag == 0) {
      if(!load(conv_weights, buffer_weights, N, lower, range)) {
	printf("Please use the MPI Weight generator to build the weights for this anisotropic function\n");
	exit(1);
      }
    }
    else {
      printf("Please use the MPI Weight generator to build the weights for this anisotropic function\n");
      exit(1);
    }
  }
}

/*
 * loads stored weights into conv_weights array
 * args:
 *      conv_weights: 2d array of weights
 *      buffer_weights: name of file where weights are stored
 *      N: number of velocity points 
 * output:
 *      1 if successful, 0 if file not found (exits if error reading a file that exists, might change later)
 */
int load(double **conv_weights, char buffer_weights[100], int N, int lower, int range) {
  FILE* fidWeights;
  int readFlag;

  if ((fidWeights = fopen(buffer_weights, "r"))) {
    printf("Loading weights from file %s\n", buffer_weights);
	fseek(fidWeights,sizeof(double)*N*N*N*lower,SEEK_SET);
    for (int i = 0; i < range; i++) {
      readFlag = fread(conv_weights[i], sizeof(double), N * N * N, fidWeights);
      if (readFlag != N * N * N) {
	printf("Error reading weight file\n");
	exit(1);
      }
    }
    fclose(fidWeights);
    return 1;
  }
  // not sure if I need to close fidWeights if failure
  return 0;
}

/*
 * writes conv_weights into file
 * args:
 *      conv_weights: 2d array of weights
 *      buffer_weights: name of file where weights are stored
 *      N: number of velocity points 
 * output:
 *      void: simply exits if failure to store
 */
void write_weights(double **conv_weights, char buffer_weights[100], int N,int lower,int range,int size,int rank) {
  double* conv_weights_buffer=(double*)malloc(sizeof(double)*N*N*N);
  int expected_size;
  MPI_Status *status;
  if(rank == 0) {
  FILE* fidWeights = fopen(buffer_weights, "w");
  
  for(int r = 1; r < size; r++){
	MPI_Recv(&expected_size,1,MPI_INT,r,0,MPI_COMM_WORLD,status);
  for (int i = 0; i < expected_size; i++) {
	MPI_Recv(conv_weights_buffer,N*N*N,MPI_DOUBLE,r,i+1,MPI_COMM_WORLD,status);
    fwrite(conv_weights_buffer, sizeof(double), N * N * N, fidWeights);
  }
  if (fflush(fidWeights) != 0) {
    printf("Something is wrong with storing the weights");
    exit(0);
  }
  free(conv_weights_buffer);
  fclose(fidWeights);
  }} else {
	MPI_Send(&range,1,MPI_INT,0,0,MPI_COMM_WORLD);
	for(int i = 0; i < range; i++){
	MPI_Send(conv_weights[i],N*N*N,MPI_DOUBLE,0,i+1,MPI_COMM_WORLD);
	}
  }	
}

/*
 *  * writes conv_weights into file
 *   * args:
 *    *      conv_weights: 2d array of weights
 *     *      buffer_weights: name of file where weights are stored
 *      *      N: number of velocity points 
 *       * output:
 *        *      void: simply exits if failure to store
 *         */
void write_zeroD(double **conv_weights, char buffer_weights[100], int N) {
  FILE* fidWeights = fopen(buffer_weights, "w");
  for (int i = 0; i < N * N * N; i++) {
    fwrite(conv_weights[i], sizeof(double), N * N * N, fidWeights);
  }
  if (fflush(fidWeights) != 0) {
    printf("Something is wrong with storing the weights");
    exit(0);
  }
  fclose(fidWeights);
}

void dealloc_weights(int range, double **conv_weights) {
  for (int i = 0; i < range; i++) {
    free(conv_weights[i]);
  }
  free(conv_weights);
}

double sinc(double x) {
  if (x != 0.0){
    return sin(x) / x;
  }
  else {
    return 1.0;
  }
}

/*
function ghat
-------------
integrated function for each convolution weight
inputs
r: integration variable
args[0]: 1/2 |zeta|
args[1]: |ki|
args[2]: |ki - 0.5*zeta|
*/
double ghat(double r, void *args) {
  double *dargs = (double *)args;

  return  pow(r, lambda + 2) * (sinc(r * dargs[0]) * sinc(r * dargs[2]) - sinc(r * dargs[1]));
}

double func_cos(double x) {
  if (fabs(x) > 1e-12) {
    return (cos(x) - 1.0) / (x * x);
  }
  else {
    return -0.5 + x * x / 24.0;
  }
}

/*
function gHat3
--------------
computes integral for each convolution weight using gauss-legendre quadrature
inputs
ki, zeta: wavenumbers for the convolution weight
*/


double gHat3(double ki1, double ki2, double ki3, double zeta1, double zeta2, double zeta3) {
  double result = 0.0;
  double error;
  gsl_function F_ghat;
  double args[3];
  gsl_integration_workspace *w_r;


  //double mu = 0.5;
  double mu = mass_j / (mass_i + mass_j);

  args[0] = mu * sqrt(zeta1 * zeta1 + zeta2 * zeta2 + zeta3 * zeta3);
  args[1] = sqrt(ki1 * ki1 + ki2 * ki2 + ki3 * ki3);
  args[2] = sqrt( (ki1 - mu * zeta1) * (ki1 - mu * zeta1) + (ki2 - mu * zeta2) * (ki2 - mu * zeta2) + (ki3 - mu * zeta3) * (ki3 - mu * zeta3) );



  w_r = gsl_integration_workspace_alloc(10000);

  F_ghat.function = &ghat;
  F_ghat.params = args;

  gsl_integration_qag(&F_ghat, 0.0, L_v, 1e-8, 1e-8, 10000, 2, w_r, &result, &error);
  //result = gauss_legendre(64,ghat,args,0.,L_v);

  gsl_integration_workspace_free(w_r);

  return prefactor * result;
}

//this generates the convolution weights G
void generate_conv_weights_iso(int lower, int range, double **conv_weights)
{
  int t,i,j,k,l,m,n;
  int index;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

#pragma omp parallel for private(index,t,l,m,n)
  for (t = 0; t < range; t++) {
	for (l = 0; l < N; l++) {
	  for (m = 0; m < N; m++) {
	    for (n = 0; n < N; n++) {

		  index = lower+t;
		  i = index/(N*N);
		  j = (index - (i*N*N))/(N);
		  k = (index - (i*N*N) - (j*N));
	
		//printf("rank %d : %d %d %d %d\n",rank,t,l,m,n);
		//fflush(stdout);
	      conv_weights[t][n + N * (m + N * l)] = wtN[l] * wtN[m] * wtN[n] * 0.25 * pow(0.5 * (diam_i + diam_j), 2) * gHat3(zeta[l], zeta[m], zeta[n], zeta[i], zeta[j], zeta[k]);
		
	    }
	  }
	}
  }
}
