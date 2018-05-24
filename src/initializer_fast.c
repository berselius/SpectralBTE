#include <math.h>
#include <stdlib.h>
#include "initializer_fast.h"
#include "weights.h"
#include "collisions_fast.h"
#include "momentRoutines.h"
#include "species.h"

/*Initializer for space homogeneous problem*/

void allocate_hom(int N, double **v, double **zeta, double **f, double **Q) {
  *v = (double *)malloc(N*sizeof(double));
  *zeta = (double *)malloc(N*sizeof(double));
  *f = (double *)malloc(N*N*N*sizeof(double));
  *Q = (double *)malloc(N*N*N*sizeof(double));
}

void initialize_hom(int N, double L_v, double *v, double *zeta, double *f, double *Q, int initFlag, int isoFlag, double lambda, double *M) {
  int i, j, k;	
  double BKWt, Temp, K;
  double dv, deta, L_eta;

  double sigma;	
  double S;
  double pre;

  printf("Initializing...%d\n",initFlag);

  /*Allocate things*/
  dv = 2*L_v / (N-1);
  //dv = 2*L_v / N;

  for(i=0;i<N;i++) {
    v[i] = -L_v + i*dv;
  }

  //Set up Fourier space grid
  deta = (2*M_PI/N) / dv;
  if( (N % 2) == 0)
    L_eta = 0.5*N*deta;
  else
    L_eta = 0.5*(N-1)*deta;

  for (i=0;i<N;i++) {
    zeta[i] = -L_eta + i*deta;
    //zeta[i] = i*deta;
  }

  /*Call initializers for other modules*/
  initialize_coll(N,L_v,v,M,zeta,lambda);

  //Initialize the moment routines
  initialize_moments_fast(N, L_v, v);

  switch(initFlag) {
    
    //Shifted isotropic problem
  case 0:        	
    printf("Isotropic problem\n");
    sigma = 0.3*L_v;	
    S = 10.0;
    for(j=0;j<N;j++) 
      for(k=0;k<N;k++) 
	for(i=0;i<N;i++) 
	  f[k + N*(j + N*i)] = exp(-1*S*(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]) - sigma)*(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]) - sigma)/(sigma*sigma))/(S*S);
    
    break;
    //Discontinuous Maxwellian problem
  case 1:
    printf("Discont problem\n");
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	for(i=0;i<N/2;i++) {
	  f[k + N*(j + N*i)] = exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]))/(0.5*M_PI*sqrt(0.5*M_PI));
	}
	for(i=N/2;i<N;i++) {
	  f[k + N*(j + N*i)] = exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]))/(M_PI*sqrt(M_PI));
	}
      }
    for(i=0;i<N;i++) {
      printf("%d %le\n",i,f[N/2 + N*(N/2 + N*i)]);
    }
    fflush(stdout);

    break;
    //BKW type problem
  case 2:
    printf("BKW\n");
    BKWt = 5.5;
    K = 1 - exp(-BKWt/6.0);
    Temp = 1.0;
    for(i=0;i<N;i++)
      for(j=0;j<N;j++)
	for(k=0;k<N;k++)
	  {
	    f[k + N*(j + N*i)] = (exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/(2*K*Temp*Temp)))/(2.0*pow(2*M_PI*K*Temp*Temp,1.5)) * ( (5*K - 3)/K + 																  (1-K)*(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/(K*K*Temp*Temp));
	  }
    
    break;
    //Two Maxwellians
  case 3:
    printf("TwoMax\n");
    sigma = M_PI/10.0;	
    pre = 0.5 / pow(2.0 * M_PI * sigma*sigma, 1.5);	
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) 
	for(i=0;i<N;i++) 
	  f[k + N*(j + N*i)] = pre*(exp(-((v[i] - 2.0*sigma)*(v[i]-2.0*sigma) + v[j]*v[j] + v[k]*v[k])/(2.0*sigma*sigma)) + exp(-((v[i] + 2*sigma)*(v[i] + 2*sigma) + v[j]*v[j] + v[k]*v[k])/(2.0*sigma*sigma)));
      
    break;
    //Simple maxwellian
  case 4:
    printf("SimpleMax\n");
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) 
	for(i=0;i<N;i++) 
	  f[k + N*(j + N*i)] = exp(-((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k])/2)/(2*M_PI*sqrt(2*M_PI));
    break;

  case 5:
    printf("TopHat\n");
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) 
	for(i=0;i<N;i++) 
	  if((fabs(v[i]) > 0.5) || (fabs(v[j]) > 0.5) || (fabs(v[k]) > 0.5))
	    f[k + N*(j + N*i)] = 0;
	  else
	    f[k + N*(j + N*i)] = 1;
    break;
  }
}
