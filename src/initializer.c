#include <math.h>
#include <stdlib.h>
#include "initializer.h"
#include "weights.h"
#include "collisions.h"
#include "momentRoutines.h"
#include "transportroutines.h"
#include "mesh_setup.h"
#include "restart.h"
#include <mpi.h>
#include "species.h"
#include <string.h>
#include "constants.h"

static double KB;
static int numspec;

/*allocator for space homogeneous problem*/

void allocate_hom(int N, double **v, double **zeta, double ***f, double ***f1, double ***Q, int num_species) {
  int i;

  numspec = num_species;

  *v = (double *)malloc(N*sizeof(double));
  *zeta = (double *)malloc(N*sizeof(double));

  *Q = malloc(num_species*num_species*sizeof(double *));
  for(i=0;i<num_species*num_species;i++)
    (*Q)[i] = malloc(N*N*N*sizeof(double));

  *f = malloc(num_species*sizeof(double *));
  for(i=0;i<num_species;i++)
    (*f)[i] = malloc(N*N*N*sizeof(double));

  *f1 = malloc(num_species*sizeof(double *));
  for(i=0;i<num_species;i++)
    (*f1)[i] = malloc(N*N*N*sizeof(double));
}


/*Initializer for space homogeneous problem*/

void initialize_hom(int N, double L_v, double *v, double *zeta, double **f, int initFlag, species *mixture) {
  int i, j, k;
  double BKWt, Temp, K;
  double dv, deta, L_eta;

  double sigma;
  double S;
  double pre;

  int spec;


  if(strcmp(mixture[0].name,"default") != 0)
    KB = KB_in_Joules_per_Kelvin;
  else
    KB = 1.;

  double maxTemp,rho;

  printf("Initializing...%d\n",initFlag);

  /*Allocate things*/
  dv = 2*L_v / (N-1);
  //dv = 2*L_v / N;
  for(i=0;i<N;i++) {
    v[i] = -L_v + i*dv;
  }

  //Set up Fourier space grid
  deta = (2*M_PI/N) / dv;
  if((N % 2) == 0)
    L_eta = 0.5*N*deta;
  else
    L_eta = 0.5*(N-1)*deta;

  for (i=0;i<N;i++) {
    zeta[i] = -L_eta + i*deta;
    //zeta[i] = i*deta;
  }

  /*Call initializers for other modules*/
  initialize_coll(N,L_v,v,zeta);

  //Initialize the moment routines
  initialize_moments(N, v, mixture);

  for(spec=0;spec<numspec;spec++) {
  switch(initFlag) {

    //Shifted isotropic problem
  case 0:
    printf("Isotropic problem\n");
    sigma = 0.3*L_v;
    S = 1.0;
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++)
	  f[spec][k + N*(j + N*i)] = exp(-1*S*(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]) - sigma)*(sqrt(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]) - sigma)/(sigma*sigma))/(S*S);

    break;


    //Discontinuous Maxwellian problem
  case 1:
    printf("Discont problem\n");
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	for(i=0;i<N/2;i++) {
	  f[spec][k + N*(j + N*i)] = exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]))/ (M_PI*sqrt(M_PI));
	}
	for(i=N/2;i<N;i++) {
	  f[spec][k + N*(j + N*i)] = exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]))/ (M_PI*sqrt(M_PI))/2;
	}
      }

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
	    f[spec][k + N*(j + N*i)] = (exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/(2*K*Temp*Temp)))/(2.0*pow(2*M_PI*K*Temp*Temp,1.5)) * ( (5*K - 3)/K + 																  (1-K)*(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/(K*K*Temp*Temp));
	  }

    break;


    //Two Maxwellians
  case 3:
    printf("TwoMax\n");
    sigma = M_PI*L_v/10.0;
    pre = 0.5 / pow(2.0 * M_PI * sigma*sigma, 1.5);
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++)
	  f[spec][k + N*(j + N*i)] = pre*(exp(-((v[i] - 2.0*sigma)*(v[i]-2.0*sigma) + v[j]*v[j] + v[k]*v[k])/(2.0*sigma*sigma)) + exp(-((v[i] + 2*sigma)*(v[i] + 2*sigma) + v[j]*v[j] + v[k]*v[k])/(2.0*sigma*sigma)));

    break;


    //Simple maxwellian
  case 4:
    /*if(spec == 0) {
      maxTemp = 300;
      rho = 5.e-3;
    }
    else if(spec == 1) {
      maxTemp = 300;
      rho = 5.e-3;
      }*/
    rho = 1.0;
    maxTemp = 1.0;
    printf("SimpleMax, rho=%g, T=%g, mass=%g KB=%g \n",rho,maxTemp,mixture[spec].mass, KB);
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++)
	  f[spec][k + N*(j + N*i)] = (rho/mixture[spec].mass) * pow(0.5*mixture[spec].mass/(M_PI*KB*maxTemp),1.5)*exp(-(0.5*mixture[spec].mass/(KB*maxTemp)) *((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k]));
	  //f[spec][k + N*(j + N*i)] = pow(1.0/M_PI,1.5)*exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]));
    break;

    //Perturbed Maxwellian
  case 5:
    printf("Perturb\n");
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++) {
	  f[spec][k + N*(j + N*i)] = ( 1 + 0.1*sin(v[i]*v[i] + v[j]*v[j] + v[k]*v[k]) )*exp(-((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k]))/(M_PI*sqrt(M_PI));
	}

    break;

  case 6:
    printf("Cutout Maxwellian\n");
    double n_cutout = 1.0e23;
    double T_cutout_K = 11000.0;
    double m_cutout = 9.1e-31;
    
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++) {
	  double v2 = v[i]*v[i] + v[j]*v[j] + v[k]*v[k];
	  f[spec][k + N*(j + N*i)] = v2 * n_cutout*pow(m_cutout / (2.0 * M_PI * KB * T_cutout_K),1.5) * exp(-0.5*m_cutout*v2 / KB / T_cutout_K);
	}

    break;
    

    /*
  case 6:
    printf("Bump on Tail \n");
    double rho1 = 0.9;
    double rho2 = 0.1;
    double T1 = 1.;
    double T2 = 1.e-4;

    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	for(i=0;i<N;i++) {
	  f[spec][k + N*(j + N*i)] =
	}

    break;
    */
  }
  }
}

void dealloc_hom(double *v, double *zeta, double **f, double **Q) {
  int i,j;
  free(v);
  free(zeta);
  for(i=0;i<numspec;i++) {
    free(f[i]);
    for(j=0;j<numspec;j++)
      free(Q[i*numspec + j]);
  }
  free(f);
  free(Q);
}

/*Space inhomogeneous case*/
void allocate_inhom(int N, int nX, double **v, double **zeta, double ****f, double ****f_conv, double ****f_1, double ***Q, int Ns) {
  int i,j;

  *v = (double *)malloc(N*sizeof(double));
  *zeta = (double *)malloc(N*sizeof(double));

  *f      = (double ***)malloc(Ns*sizeof(double **));
  *f_conv = (double ***)malloc(Ns*sizeof(double **));
  *f_1    = (double ***)malloc(Ns*sizeof(double **));
  *Q = (double **)malloc(Ns*Ns*sizeof(double *));

  for(i=0;i<Ns;i++) {
    (*f)[i]      = (double **)malloc(nX*sizeof(double *));
    (*f_conv)[i] = (double **)malloc(nX*sizeof(double *));
    (*f_1)[i]    = (double **)malloc(nX*sizeof(double *));
    for(j=0;j<Ns;j++)
      (*Q)[j*Ns + i] = malloc(N*N*N*sizeof(double));
  }
}

void initialize_inhom(int N, int Ns, double L_v, double *v, double *zeta, double ***f, double ***f_conv, double ***f_1, species *mixture, int initFlag, int nX, double *xnodes, double *dxnodes, double dt, int *t, int order, int restart, char *inputfilename) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  int i,j,k,l,m;
  double dv, L_eta, deta;

  double rho_l, T_l, ux_l, rho_r, T_r, ux_r;
  double TWall = 1.0;

  printf("Initializing for inhomogeneous run\n");


  if(strcmp(mixture[0].name,"default") != 0)
    KB = KB_in_Joules_per_Kelvin;
  else
    KB = 1.;

  double Ma = 1;

  /*Initialize things*/
  dv = 2*L_v / (N-1);
  for(i=0;i<N;i++) {
    v[i] = -L_v + i*dv;
  }

  //Set up Fourier space grid
  L_eta = 0.5*(N-1)*M_PI/L_v;
  deta = M_PI*(N-1)/(N*L_v);
  for (i=0;i<N;i++) {
    zeta[i] = -L_eta + i*deta;
  }

  /*Call initializers for other modules*/
  initialize_coll(N,L_v,v,zeta);

  //Initialize the moment routines
  initialize_moments(N, v, mixture);

  //Initialize transport routines
  initialize_transport(N, nX, L_v, xnodes, dxnodes, v, initFlag, dt, 1.0, mixture);

  for(i=0;i<Ns;i++)
    for(l=0;l<(nX+2*order);l++) {
      f[i][l]      = malloc(N*N*N*sizeof(double));
      f_conv[i][l] = malloc(N*N*N*sizeof(double));
      f_1[i][l]    = malloc(N*N*N*sizeof(double));
    }

  //Initialize F based on initial conditions...

  init_restart(nX, order, N, Ns, mixture);

  if(restart == 1) {
    if(rank == 0) {
      printf("Loading from previously generated data\n");
    }
    load_restart(f,t,inputfilename);
  }
  else {
    *t = 0;

    //set some defaults
    rho_l = 1.0;
    ux_l = 0.0;
    T_l = 1.0;

    rho_r = 1.0;
    ux_r = 0.0;
    T_r = 1.0;

    switch(initFlag) {
    case 0:
      rho_l = 4.0*Ma*Ma/(Ma*Ma + 3.0);
      T_l = (5.0*Ma*Ma - 1.0)*(Ma*Ma + 3.0)/(16.0*Ma*Ma);
      ux_l = -sqrt(5.0/3.0)*(Ma*Ma + 3.0)/(4.0*Ma);

      rho_r = 1.0;
      T_r = 1.0;
      ux_r = -Ma*sqrt(5.0/3.0);
      printf("Shock problem: left rho:%g ux:%g T:%g, right rho:%g ux:%g T:%g\n",rho_l,ux_l,T_l,rho_r,ux_r,T_r);
      break;
    case 1:
      rho_r = 1.0;
      T_r = TWall;
      printf("Sudden heating problem: Initial dist rho:%g ux:0 T:%g TWall:%g\n",rho_r,T_r,2.0*T_r);
      break;
    case 2:
      rho_l = 1.0;
      T_r = 2.0;
      T_l = 1.0;
      ux_l = -1.0;
      ux_r = -1.0;
      printf("Two Maxwellian problem?: rho:%g, ux_l:%g ux_r:%g T_l:%g, T_r:%g\n",rho_l,ux_l,ux_r,T_l, T_r);
      break;
    case 3:
      //Tr = T0 + x[l]*(T1-T0);
      rho_r = 1.0;
      T_r = 1.5;
      printf("Heat transfer problem: rho:%g TAvg:%g\n",rho_r,T_r);
      break;
    case 5:
      rho_l = 1.0;
      T_l = 1.0;
      printf("Poiseuille: rho%g T%g\n",rho_l,T_l);
      break;
    case 6:
      rho_l = 1.0;
      ux_l = 1.2972;
      T_l = 1.0;

      rho_r = 1.297;
      ux_r = 1.0;
      T_r = 1.195;

      printf("Shock problem: left rho:%g ux:%g T:%g, right rho:%g ux:%g T:%g\n",rho_l,ux_l,T_l,rho_r,ux_r,T_r);
      break;
    }

    double maxTemp = T_r;

    for(m=0;m<Ns;m++) {
      printf("%d %s %g\n",m, mixture[m].name, mixture[m].mass);

    for(l=order;l<(nX+order);l++) {
      switch (initFlag) {
      case 0:
	if(l < nX/2)  {
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++)
	      for(k=0;k<N;k++) {
		f[m][l][k + N*(j + N*i)] = rho_l*exp(-((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k])/T_l)/((T_l*M_PI)*sqrt(T_l*M_PI));
	      }
	}
	else {
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++)
	      for(k=0;k<N;k++) {
		f[m][l][k + N*(j + N*i)] = rho_r*exp(-((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k])/T_r)/((T_r*M_PI)*sqrt(T_r*M_PI));
	      }
	}
	break;
      case 1:
	for(i=0;i<N;i++)
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++) {
	      f[m][l][k + N*(j + N*i)] = (rho_r/mixture[m].mass) * pow(0.5*mixture[m].mass/(M_PI*KB*maxTemp),1.5)*exp(-(0.5*mixture[m].mass/(KB*maxTemp)) *((v[i])*(v[i]) + v[j]*v[j] + v[k]*v[k]));
	    }
	break;
      case 2:
	for(i=0;i<N;i++)
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++) {
	      f[m][l][k + N*(j + N*i)] = rho_l*exp(-((v[i] - ux_l)*(v[i] - ux_l) + v[j]*v[j] + v[k]*v[k])/T_l)/((T_l*M_PI)*sqrt(T_l*M_PI));
	    }
	break;
      case 3:
	for(i=0;i<N;i++)
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++) {
	      f[m][l][k + N*(j + N*i)] = rho_r*exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/T_r)/((T_r*M_PI)*sqrt(T_r*M_PI));
	    }
	break;
      case 5:  //Poiseuille
	for(i=0;i<N;i++)
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++) {
	      f[m][l][k + N*(j + N*i)] = rho_l*exp(-(v[i]*v[i] + v[j]*v[j] + v[k]*v[k])/T_l)/((T_l*M_PI)*sqrt(T_l*M_PI));
	    }
	break;
      case 6: //Mach 1.2 shock
	if(l < nX/2)  {
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++)
	      for(k=0;k<N;k++) {
		f[m][l][k + N*(j + N*i)] = rho_l*exp(-((v[i] - ux_l)*(v[i]-ux_l) + v[j]*v[j] + v[k]*v[k])/T_l)/((T_l*M_PI)*sqrt(T_l*M_PI));
	      }
	}
	else {
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++)
	      for(k=0;k<N;k++) {
		f[m][l][k + N*(j + N*i)] = rho_r*exp(-((v[i]-ux_r)*(v[i]-ux_r) + v[j]*v[j] + v[k]*v[k])/T_r)/((T_r*M_PI)*sqrt(T_r*M_PI));
	      }
	}
	break;
      }
    }

    }
  }

}

void dealloc_inhom(int nX, int order, double *v, double *zeta, double ***f, double ***f_conv, double ***f_1, double **Q) {
  int i,j,l;

  free(v);
  free(zeta);
  for(i=0;i<numspec;i++) {
    for(j=0;j<numspec;j++)
      free(Q[j*numspec + i]);
    for(l=0;l<nX+2*order;l++) {
      free(f[i][l]);
      free(f_conv[i][l]);
      free(f_1[i][l]);
    }
    free(f[i]);
    free(f_conv[i]);
    free(f_1[i]);
  }
  free(Q);
  free(f);
  free(f_conv);
}
