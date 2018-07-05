//This manages all of the outputting from the Boltzmann code

//Format of the output files will be set up for something numpy can load, I hope. columns are:

// Time
// Spatial location (just 0 for a 0D problem)
// Mass density
// Bulk Velocity
// Temperature
// Distribution slice, if on

#include "output.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "momentRoutines.h"
#include "input.h"
#include <mpi.h>
#include "species.h"
#include "constants.h"

/*********************
MPI STUFF
 *********************/
static int rank;
static int numNodes;

//File streams for output
static FILE **fidOutput;
static FILE **fidRho;
static FILE **fidKinTemp;
static FILE **fidPressure;
static FILE **fidMarginal;
static FILE **fidSlice;
static FILE **fidBulkV;

//Parameters from main code
static int N;
static double L_v;
static int nX;
static int nX_node;
static double *x, *dx;

//flags for output, default set to output them
static int densFlag = 1;
static int velFlag = 1;
static int tempFlag = 1;
static int presFlag = 1;
static int marginalFlag = 1;
static int sliceFlag = 1;
static int entFlag = 1;

static species *mixture;
static int Ns;

static void fopen_output_file(const char* format, char* inputFile,species* mixture, const char* restartOpt);
static void read_flags(FILE* outParams);

//Takes the Linf norm of the difference with maxwellian
double MaxConverge(int N, double *v, double rho, double* u, double T, double *f) {
    double max = 0, test = 0;
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                test = fabs(f[k + N * (j + N * i)] - (rho / pow(2 * M_PI * T, 1.5)) * exp(-( (v[i] - u[0]) * (v[i] - u[0]) + (v[j] - u[1]) * (v[j] - u[1]) + (v[k] - u[2]) * (v[k] - u[2])) / (2 * T)));
                if (test > max) {
                    max = test;
                }
            }
        }
    }
    return max;
}

//initializer for homogeneous case
void initialize_output_hom(int nodes, double Lv, int restart, char *inputFile, char *outputOptions, species *mix, int num_species) {

    N = nodes;
    L_v = Lv;
    mixture = mix;
    Ns = num_species;

    char path[100] = {"./input/"};

    strcat(path, outputOptions);

    //Load flag data from file
    FILE *outParams = fopen(path, "r");

    //Tells the stream writers which things to output
    read_flags(outParams);

    fclose(outParams);

    printf("Opening output files \n");

    const char* format = "Data/moments_";

    const char* option = NULL;

    if(restart) 
      option = "a";
    else
      option = "w";

    fopen_output_file(format, inputFile, mixture, option);    
}


static void fopen_output_file(const char* format, char* inputFile, species* mixture, const char* restartOpt){

    fidOutput = malloc(Ns * sizeof(FILE *));

    char buffer[256];

    for(int i = 0; i < Ns; i++) {
        sprintf(buffer, format, inputFile, mixture[i].name);
	fidOutput[i] = fopen(buffer, restartOpt);
    }
}



//This version is for homogeneous case
void write_streams(double **f, double time) {
    int i, l;
    double density, kinTemp, bulkV[3], energy[2];
    
    //We will print everything to this buffer
    char line[1028];
    char buff[32];

    for (i = 0; i < Ns; i++) {

      //Get moments from the species

      density = getDensity(f[i], 0);
      if (isnan( density)) {
	printf("nan detected\n");
	exit(0);
      }
      getBulkVelocity(f[i], bulkV, density, 0);
      kinTemp = getTemperature(f[i], bulkV, density, 0);
      getEnergy(f[i], energy);
      
      sprintf(line, "%le", time);

      //Build the output line
      if (densFlag) {	
	sprintf(buff, "%le", density);      
	strcat(line,buff);
      }
      if (velFlag) {
	sprintf(buff, "%le", bulkV[0]);
	strcat(line,buff);
      }
      if (tempFlag) {
	sprintf(buff, "%le", kinTemp);
	strcat(line,buff);
      }
      if (presFlag) {
	sprintf(buff, "%le", getPressure(density,kinTemp));
	strcat(line,buff);
      }
      if (entFlag) {
	sprintf(buff, " %le", energy[1] / energy[0]);
	strcat(line,buff);
      }
      if (sliceFlag) {
	for (l = 0; l < N; l++) {
	  sprintf(buff, " %le", f[i][N / 2 + N * (N / 2 + N * l)]);
	  strcat(line,buff);
	}
      }
      strcat(line,"\n");
      fprintf(fidOutput[i],"%s",line);
    }
}

void initialize_output_inhom(int nodes, double Lv, int numX, int numX_node, double *xnodes, double *dxnodes, int restart, char *inputFile, char *outputOptions, species *mix, int num_species) {

    N = nodes;
    L_v = Lv;
    mixture = mix;
    Ns = num_species;

    nX = numX;
    nX_node = numX_node;
    x = xnodes;
    dx = dxnodes;
    
    char path[100] = {"./input/"};

    strcat(path, outputOptions);

    //Load flag data from file
    FILE *outParams = fopen(path, "r");

    //Sets flags so the stream writer knows what to output
    read_flags(outParams);

    fclose(outParams);

    //Set this up to have all ranks send moments to rank 0 for output
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Opening output files \n");

	const char* format = "Data/moments_";	

        const char* option = NULL;

        // I dont know if this will work
        if(restart){
            option = "a";
        } else {
            option = "w";
        }

	fopen_output_file(format, inputFile, mixture, option);
    }
}

void write_streams_inhom(double ***f, double time, double *v, int order) {
    int i, j, k, l, spec;
    double density, kinTemp, bulkV[3], marg, h_v;

    h_v = 2 * L_v / N;

    MPI_Status status;
    double momentBuffer[6];

    if (rank == 0) {
        for (spec = 0; spec < Ns; spec++) {
            if (densFlag)
                fprintf(fidRho[spec], "ZONE I=%d T=\"T=%g\"\n", nX, time);
            if (tempFlag)
                fprintf(fidKinTemp[spec], "ZONE I=%d T=\"T=%g\"\n", nX, time);
            if (presFlag)
                fprintf(fidPressure[spec], "ZONE I=%d T=\"T=%g\"\n", nX, time);
            if (velFlag)
                fprintf(fidBulkV[spec], "ZONE I=%d T=\"T=%g\"\n", nX, time);

            for (l = order; l < (nX_node + order); l++) {
                density = getDensity(f[spec][l], 0);
                if (isnan( density)) {
                    printf("nan detected in rank 0 cell %d \n", l);
                    exit(0);
                }
                getBulkVelocity(f[spec][l], bulkV, density, 0);
                kinTemp = getTemperature(f[spec][l], bulkV, density, 0);

                if (densFlag)
                    fprintf(fidRho[spec], "%le %le\n", x[l], density);
                if (velFlag)
                    fprintf(fidBulkV[spec], "%le %le\n", x[l], bulkV[0]);
                if (tempFlag)
                    fprintf(fidKinTemp[spec], "%le %le\n", x[l], kinTemp);
                if (presFlag)
                    fprintf(fidPressure[spec], "%le %le\n", x[l], getPressure(density, kinTemp));
                if (sliceFlag) {
                    fprintf(fidSlice[spec],    "ZONE I=%d T=\"T=%g X=%g\" \n", N, time, x[l]);
                    for (i = 0; i < N; i++)
                        fprintf(fidSlice[spec], "%le %le\n", v[i], f[spec][l][N / 2 + N * (N / 2 + N * i)]);
                    fprintf(fidSlice[spec], "\n");
                }
                if (marginalFlag) {
                    fprintf(fidMarginal[spec], "ZONE I=%d T=\"T=%g X=%g\" \n", N, time, x[l]);
                    for (i = 0; i < N; i++) {
                        marg = 0.0;
                        for (j = 0; j < N; j++)
                            for (k = 0; k < N; k++) {
                                marg += h_v * h_v * f[spec][l][k + N * (j + N * i)];
                            }

                        fprintf(fidMarginal[spec], "%le %le\n", v[i], marg);
                    }
                    fprintf(fidMarginal[spec], "\n");
                }
            }

            int nodeCounter;
            for (nodeCounter = 1; nodeCounter < numNodes; nodeCounter++) {
                for (l = order; l < nX_node + order; l++) {
                    MPI_Recv(momentBuffer, 6, MPI_DOUBLE, nodeCounter, l, MPI_COMM_WORLD, &status);
                    if (densFlag)
                        fprintf(fidRho[spec], "%le %le\n", momentBuffer[5], momentBuffer[0]);
                    if (velFlag)
                        fprintf(fidBulkV[spec], "%le %le\n", momentBuffer[5], momentBuffer[1]);
                    if (tempFlag)
                        fprintf(fidKinTemp[spec], "%le %le\n", momentBuffer[5], momentBuffer[4]);
                    if (presFlag)
                        fprintf(fidPressure[spec], "%le %le\n", momentBuffer[5], getPressure(momentBuffer[0], momentBuffer[4]));
                    //if(sliceFlag) {
                    //fprintf(fidSlice, "ZONE I=%d T=\"T=%g X=%g\" \n", N, time, x[l]);
                    //for(i=0;i<N;i++)
                    //fprintf(fidSlice, "%le %le\n", v[i], f[l][N/2 + N*(N/2 + N*i)]);
                    //fprintf(fidSlice,"\n");
                }
            }
            if (densFlag)
                fflush(fidRho[spec]);
            if (velFlag)
                fflush(fidBulkV[spec]);
            if (tempFlag)
                fflush(fidKinTemp[spec]);
            if (presFlag)
                fflush(fidPressure[spec]);
            if (marginalFlag)
                fflush(fidMarginal[spec]);
            if (sliceFlag)
                fflush(fidSlice[spec]);
        }
    }
    else {
        for (spec = 0; spec < Ns; spec++) {
            for (l = order; l < (nX_node + order); l++) {
                density = getDensity(f[spec][l], 0);
                if (isnan( density)) {
                    printf("nan detected in rank %d cell %d\n", rank, l);
                    //exit(0);
                }
                getBulkVelocity(f[spec][l], bulkV, density, 0);
                kinTemp = getTemperature(f[spec][l], bulkV, density, 0);
                momentBuffer[0] = density;
                momentBuffer[1] = bulkV[0];
                momentBuffer[2] = bulkV[1];
                momentBuffer[3] = bulkV[2];
                momentBuffer[4] = kinTemp;
                momentBuffer[5] = x[l];

                MPI_Send(momentBuffer, 6, MPI_DOUBLE, 0, l, MPI_COMM_WORLD);
            }
        }
    }
}


void close_streams() {

    for (int i = 0; i < Ns; i++) {
      fclose(fidOutput[i]);        
    }

    free(fidOutput);
}


static void read_flags(FILE* outParams) {
    char line[80] = {"dummy"};

    read_line(outParams, line);

    /*Read file*/
    while (strcmp(line, "Stop") != 0) {
        // read one line at time 
        read_line(outParams, line);

        if (strcmp(line, "density") == 0) {
            densFlag = read_int(outParams);
        } else if (strcmp(line, "velocity") == 0) {
            velFlag = read_int(outParams);
        } else if(strcmp(line, "temperature") == 0) {
            tempFlag = read_int(outParams);
        } else if (strcmp(line, "pressure") == 0) {
            presFlag = read_int(outParams);
        } else if (strcmp(line, "marginal") == 0) {
            marginalFlag = read_int(outParams);
        } else if (strcmp(line, "slice") == 0) {
            sliceFlag = read_int(outParams);
        } if (strcmp(line, "entropy") == 0) {
            entFlag = read_int(outParams);
        }
    }
}
