//This manages all of the outputting from the Boltzmann code

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

//Parameters from main code
static int N;
static double L_v;
static double *v;
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
void initialize_output_hom(int nodes, double Lv, int restart, char *inputFile, char *outputOptions, species *mix, double *velo, int num_species) {

    N = nodes;
    L_v = Lv;
    mixture = mix;
    Ns = num_species;
    v = velo;

    char path[100] = {"./input/"};
    char topline[1028];
    char tmpbuff[32];

    strcat(path, outputOptions);

    //Load flag data from file
    FILE *outParams = fopen(path, "r");

    //Tells the stream writers which things to output
    read_flags(outParams);

    fclose(outParams);

    printf("Opening output files \n");

    const char* prefac = "Data/moments_";

    const char* option = NULL;

    if(restart) 
      option = "a";
    else
      option = "w";

    fopen_output_file(prefac, inputFile, mixture, option);    
    

    //Print headers
    for(int i=0; i<Ns; i++) {
      sprintf(topline,"#");
      
      sprintf(tmpbuff,"Time ");
      strcat(topline,tmpbuff);

      if(densFlag) {
	sprintf(tmpbuff,"Density ");
	strcat(topline,tmpbuff);
      }
      if(velFlag) {
	sprintf(tmpbuff,"Velocity_x ");
	strcat(topline,tmpbuff);
      }
      if(tempFlag) {
	sprintf(tmpbuff,"Temperature ");
	strcat(topline,tmpbuff);
      }
      if(presFlag) {
	sprintf(tmpbuff,"Pressure ");
	strcat(topline,tmpbuff);
      }
      if(entFlag) {
	sprintf(tmpbuff,"Entropy");
	strcat(topline,tmpbuff);
      }
      if(sliceFlag) {
	for(int l=0;l<N;l++) {
	  sprintf(tmpbuff,"v:%le ",v[l]);
	  strcat(topline,tmpbuff);
	}
      }
      strcat(topline,"\n");
      fprintf(fidOutput[i],topline);
    }
}


static void fopen_output_file(const char* prefac, char* inputFile, species* mixture, const char* restartOpt){

    fidOutput = malloc(Ns * sizeof(FILE *));

    char buffer[256];

    for(int i = 0; i < Ns; i++) {
      sprintf(buffer, prefac);
      strcat(buffer,inputFile);
      if(Ns > 1) {
	strcat(buffer,"_");
	strcat(buffer,mixture[i].name);
      }
      fidOutput[i] = fopen(buffer, restartOpt);
    }
}



//This version is for homogeneous case
void write_streams(double **f, double time) {
    int i, l;
    double density, kinTemp, bulkV[3], entropy;
    
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
      entropy = getEntropy(f[i]);
      
      sprintf(line, "%le ", time);

      //Build the output line
      if (densFlag) {	
	sprintf(buff, "%le ", density);      
	strcat(line,buff);
      }
      if (velFlag) {
	sprintf(buff, "%le ", bulkV[0]);
	strcat(line,buff);
      }
      if (tempFlag) {
	sprintf(buff, "%le ", kinTemp);
	strcat(line,buff);
      }
      if (presFlag) {
	sprintf(buff, "%le ", getPressure(density,kinTemp));
	strcat(line,buff);
      }
      if (entFlag) {
	sprintf(buff, " %le ", entropy);
	strcat(line,buff);
      }
      if (sliceFlag) {
	for (l = 0; l < N; l++) {
	  sprintf(buff, " %le ", f[i][N / 2 + N * (N / 2 + N * l)]);
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
    char topline[1028];
    char tmpbuff[32];

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

	//Print headers
	for(int i=0; i<Ns; i++) {
	  sprintf(topline,"#");
	  
	  sprintf(tmpbuff,"Time ");
	  strcat(topline,tmpbuff);

	  sprintf(tmpbuff,"Position ");
	  strcat(topline,tmpbuff);
	  
	  if(densFlag) {
	    sprintf(tmpbuff,"Density ");
	    strcat(topline,tmpbuff);
	  }
	  if(velFlag) {
	    sprintf(tmpbuff,"Velocity_x ");
	    strcat(topline,tmpbuff);
	  }
	  if(tempFlag) {
	    sprintf(tmpbuff,"Temperature ");
	    strcat(topline,tmpbuff);
	  }
	  if(presFlag) {
	    sprintf(tmpbuff,"Pressure ");
	    strcat(topline,tmpbuff);
	  }

	  strcat(topline,"\n");
	  fprintf(fidOutput[i],topline);
	}
	
    }
}

void write_streams_inhom(double ***f, double time, int order) {
    int l, spec;
    double density, kinTemp, bulkV[3];

    MPI_Status status;
    double momentBuffer[6];

    char line[1028];
    char buff[32];


    if (rank == 0) {
        for (spec = 0; spec < Ns; spec++) {
            for (l = order; l < (nX_node + order); l++) {

	      //get the moments
	      density = getDensity(f[spec][l], 0);
	      if (isnan( density)) {
		printf("nan detected in rank 0 cell %d \n", l);
		exit(0);
	      }
	      getBulkVelocity(f[spec][l], bulkV, density, 0);
	      kinTemp = getTemperature(f[spec][l], bulkV, density, 0);
	      
	      
	      sprintf(line, "%le ", time);
	      
	      sprintf(buff, "%le ", x[l]);      
	      strcat(line,buff);
	      
	      //Build the output line
	      if (densFlag) {	
		sprintf(buff, "%le ", density);      
		strcat(line,buff);
	      }
	      if (velFlag) {
		sprintf(buff, "%le ", bulkV[0]);
		strcat(line,buff);
	      }
	      if (tempFlag) {
		sprintf(buff, "%le ", kinTemp);
		strcat(line,buff);
	      }
	      if (presFlag) {
		sprintf(buff, "%le ", getPressure(density,kinTemp));
		strcat(line,buff);
	      }
	      strcat(line,"\n");
	      fprintf(fidOutput[spec],"%s",line);	      
            }


	    //Next, get moment data from all the other nodes in order
	    //Note: this was implemented in a SUPER clunky way
            int nodeCounter;
            for (nodeCounter = 1; nodeCounter < numNodes; nodeCounter++) {
	      for (l = order; l < nX_node + order; l++) {
		MPI_Recv(momentBuffer, 6, MPI_DOUBLE, nodeCounter, l, MPI_COMM_WORLD, &status);		

		sprintf(line, "%le ", time);
		
		sprintf(buff, "%le ", momentBuffer[5]);      
		strcat(line,buff);
		
		//Build the output line
		if (densFlag) {	
		  sprintf(buff, "%le ", momentBuffer[0]);      
		  strcat(line,buff);
		}
		if (velFlag) {
		  sprintf(buff, "%le ", momentBuffer[1]);
		  strcat(line,buff);
		}
		if (tempFlag) {
		  sprintf(buff, "%le ", momentBuffer[4]);
		  strcat(line,buff);
		}
		if (presFlag) {
		  sprintf(buff, "%le ", getPressure(momentBuffer[0],momentBuffer[4]));
		  strcat(line,buff);
		}
		strcat(line,"\n");
		fprintf(fidOutput[spec],"%s",line);	      		
	      }
            }
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
