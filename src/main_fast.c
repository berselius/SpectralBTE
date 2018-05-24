#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fftw3.h>
#include "initializer_fast.h"
#include "input.h"
#include "output.h"
#include "collisions_fast.h"
#include <omp.h>
#include <string.h>
#include <math.h>
#include "species.h"
#include "weights.h"

int main(int argc, char **argv) {
  //Top-level parameters, read from input
  int N;
  double L_v;
  double Kn;
  double lambda;
  double dt;
  int nT;
  int order;
  int dataFreq;
  int restart;
  double restart_time;
  int initFlag;
  int bcFlag;
  int homogFlag;
  int weightFlag;
  int isoFlag;
  char *meshFile;
  double *M;

  //other top level parameters
  double *v;
  double *zeta;
  double *f_hom;
  double *Q;

  //Species data
  int Ns;
  species *mixture;
  char **species_names;

  //command line arguments
  char *inputFilename = malloc(80*sizeof(char));
  strcpy(inputFilename,argv[1]);

  char *outputChoices = malloc(80*sizeof(char));  
  strcpy(outputChoices,argv[2]);

  //Variables for main function
  int t;
  int i,j,k;
  int outputCount;
  double t1, t2;

  printf("Reading input\n");
  read_input(&N, &L_v, &Kn, &lambda, &dt, &nT, &order, &dataFreq, &restart, &restart_time, &initFlag, &bcFlag, &homogFlag, &weightFlag, &isoFlag, &meshFile, &Ns, &species_names, inputFilename);
  
  load_and_allocate_spec(&mixture, Ns, species_names);

  if(Ns != 1) {
    printf("Error - fast method only written for one species (for now)\n");
  }

  printf("Initializing variables %d\n",homogFlag);

  M = malloc(3*sizeof(double));
  M[0] = 8;
  M[1] = 8;
  M[2] = 8;

  if(homogFlag == 0) {
    allocate_hom(N,&v,&zeta,&f_hom,&Q);
    initialize_hom(N, L_v, v, zeta, f_hom, Q, initFlag, isoFlag, lambda, M);
  }

  printf("Initializing output %s\n",outputChoices);
  initialize_output_hom(N, L_v, restart, inputFilename, outputChoices, mixture, Ns);

  species *mix = malloc(sizeof(species));
  mix[0].mass = 1.0;
  mix[0].d_ref = 2.0;
  strcpy(mix[0].name,"default");

  t = 0;
  outputCount = 0;

  for(i=0;i<N;i++)
    printf("v[%d]: %le, zeta[%d]: %le\n", i,v[i],i,zeta[i]);

  write_streams(&f_hom,0,v);
  while(t < nT) {
    printf("In step %d of %d\n",t+1,nT);

    t1 = omp_get_wtime();
    collision(f_hom,Q);
    t2 = omp_get_wtime();
    printf("Time elapsed: %g\n",t2-t1);

    for(i=0;i<N;i++) {
      for(j=0;j<N;j++)
	for(k=0;k<N;k++) {	
	  f_hom[k + N*(j + N*i)] = f_hom[k + N*(j + N*i)] + dt*Q[k + N*(j + N*i)]/Kn;
	}
    }

    outputCount++;
    if(outputCount % dataFreq == 0) {
      write_streams(&f_hom,dt*(t+1),v);
      outputCount = 0;
    }
    t = t+1;
  }


  return 0;
}



