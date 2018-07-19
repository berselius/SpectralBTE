#include "weights.h"
#include <math.h>
#include "gauss_legendre.h"
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include "species.h"
#include <string.h>
#include <stdarg.h>

static int N;
static double L_v;
static double *zeta;
static double deta;
static double prefactor;
static double lambda;
static double mass_i;
static double mass_j;
static double mu_ij;
//static gsl_integration_glfixed_table *GL_table;
//static double max = 0.0;
static double *wtN;

void alloc_weights(int N, double ****conv_weights, int total_species) {
  int i;
  *conv_weights = malloc(total_species*sizeof(double **));
  for(i=0;i<total_species;i++)
    (*conv_weights)[i] = malloc(N*N*N*sizeof(double *));
}

void initialize_weights(int nodes, double *eta, double Lv, double lam, int weightFlag, int isoFlag, species species_i, species species_j, int weightgenFlag, ...) {
  char buffer_weights[100];
  int readFlag;
  int i;

  N = nodes;
  zeta = eta;
  deta = eta[1]-eta[0];
  L_v = Lv;
  lambda = lam;
  prefactor = 16.0*M_PI*M_PI*deta*deta*deta/pow(2.0*M_PI,1.5)/(4.0*M_PI);
  diam_i = species_i.d_ref;
  diam_j = species_j.d_ref;
  mass_i = species_i.mass;
  mass_j = species_j.mass;
  mu_ij = mass_i*mass_j/(mass_i + mass_j);

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  #pragma omp parallel for
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;

  //GL_table = gsl_integration_glfixed_table_alloc(64);

  printf("%g %g %g %g %s %s \n",diam_i,diam_j,mass_i,mass_j,species_i.name,species_j.name);

  if (weightgenFlag == 0) {
    double **conv_weights;
    va_list args;
    va_start(args, weightgenFlag);
    conv_weights = va_arg(args, double **);
    va_end(args);

    FILE *fidWeights;

    printf("Precomputing weights to be stored...\n");
    for(i=0;i<N*N*N;i++) {
      conv_weights[i] = malloc(N*N*N*sizeof(double));
    }

    if(!isoFlag) {
      if(strcmp(species_i.name,"default")==0)
        sprintf(buffer_weights,"Weights/N%d_isotropic_L_v%g_lambda%g.wts",N, L_v,lambda); //old style of naming
      else
        sprintf(buffer_weights,"Weights/N%d_isotropic_L_v%g_HS_%s_%zd_%s_%zd.wts",N, L_v,species_i.name,species_i.id,species_j.name,species_j.id);
    }
    else {
      sprintf(buffer_weights,"Weights/N%d_AnIso_L_v%g_lambda%g_Landau.wts",N, L_v,lambda);
      //sprintf(buffer_weights,"Weights/N%d_AnIso_L_v%g_lambda%g_glance0.0001_C.wts",N, L_v,lambda);
    }


    if(weightFlag == 0) { //Check to see if the weights are there
      if((fidWeights = fopen(buffer_weights,"r"))) {
        printf("Loading weights from file %s\n",buffer_weights);
        for(i=0;i<N*N*N;i++) { 
          readFlag = (int) fread(conv_weights[i],sizeof(double),N*N*N,fidWeights);
          if(readFlag != N*N*N) {
	    printf("Error reading weight file\n");
            exit(1);
          } 
        }      
      }
      else {
        printf("Stored weights not found for this configuration, generating ...\n");
        if(!isoFlag) {
          generate_conv_weights_iso(conv_weights);
        }
        else {
          printf("Please use the MPI Weight generator to build the weights for this anisotropic function\n");
          exit(1);
        }
        //dump the weights we've computed into a file
      
        fidWeights = fopen(buffer_weights,"w");
        for(i=0;i<N*N*N;i++) {
          fwrite(conv_weights[i],sizeof(double),N*N*N,fidWeights);
        } 
        if(fflush(fidWeights) != 0) {
          printf("Something is wrong with storing the weights");
          exit(0);
        }      
      }
      fclose(fidWeights);
    }
    else { //weights forced to be regenerated
      printf("Fresh version of weights being computed and stored for this configuration\n");
      generate_conv_weights_iso(conv_weights);
      //dump the weights we've computed into a file
      fidWeights = fopen(buffer_weights,"w");
      for(i=0;i<N*N*N;i++) {
        fwrite(conv_weights[i],sizeof(double),N*N*N,fidWeights);
      } 
      if(fflush(fidWeights) != 0) {
        printf("Something is wrong with storing the weights");
        exit(0);
      }
      fclose(fidWeights);
    }
  }
  else {
    printf("Not precomputing weights. The weights will be generated on-the-fly...\n");
  }
}

void dealloc_weights(int N, double **conv_weights) {
  int i;
  for(i=0;i<N*N*N;i++) {
    free(conv_weights[i]);
  }
  free(conv_weights);
}


double sinc(double x) {
  double res;
  if (x != 0.0)
    res = sin(x) / x;
  else
    res = 1.0;
  return res;
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

  return  pow(r,lambda+2)*(sinc(r*dargs[0])*sinc(r*dargs[2]) - sinc(r*dargs[1]));
}


double func_cos(double x) {
  double res;
  if(fabs(x) > 1e-12) 
    res = (cos(x) - 1.0)/(x*x);
  else
    res = -0.5 + x*x/24.0;
  return res;
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
  
  args[0] = mu*sqrt(zeta1*zeta1 + zeta2*zeta2 + zeta3*zeta3);
  args[1] = sqrt(ki1*ki1 + ki2*ki2 + ki3*ki3);
  args[2] = sqrt( (ki1 - mu*zeta1)*(ki1 - mu*zeta1) + (ki2 - mu*zeta2)*(ki2 - mu*zeta2) + (ki3 - mu*zeta3)*(ki3 - mu*zeta3) );
 

  w_r = gsl_integration_workspace_alloc(10000);
  
  F_ghat.function = &ghat;
  F_ghat.params = args;
  gsl_integration_qag(&F_ghat, 0.0, L_v, 1e-8,1e-8,10000,2,w_r,&result,&error);
  //result = gauss_legendre(64,ghat,args,0.,L_v);
  gsl_integration_workspace_free(w_r);
  
  
  /*          
    int origFlag = 0;
    int origFlag2 = 0;

    double ap = (args[0] + args[2])*L_v;
    double am = (args[0] - args[2])*L_v;

  if(lambda == 1) {

    //GAIN


    if(fabs(args[0] - args[2]) < 1e-10)
      am = 0.0;

    double xL = L_v*args[0];
    double yL = L_v*args[2];

    if((args[0] > 1e-8) && (args[2] > 1e-8)) {
      origFlag = 1;
      gain = (L_v*L_v*0.5/(args[0]*args[2])) * (sinc(am) - sinc(ap) + func_cos(am) - func_cos(ap));
    }
    else if ((args[0] > 1e-8) && (args[2] < 1e-8)) {
      gain = L_v*(sin(xL) - xL*cos(xL)          )*pow(args[0],-3) + L_v*args[2]*args[2]*(xL*(xL*xL - 6.0)*cos(xL)  - 3.0*(xL*xL - 2.0)*sin(xL)       )*pow(args[0],-5)/6.0
	       + (xL*sin(xL) + 2.0*cos(xL) - 2.0)*pow(args[0],-4) -     args[2]*args[2]*(xL*(xL*xL - 18.0)*sin(xL) + 6.0*(xL*xL - 4.0)*cos(xL) + 24.0)*pow(args[0],-6)/6.0; 
    }
    else if ((args[2] > 1e-8) && (args[0] < 1e-8)) {
      gain = L_v*(sin(yL) - yL*cos(yL)          )*pow(args[2],-3) + L_v*args[0]*args[0]*(yL*(yL*yL - 6.0)*cos(yL)  - 3.0*(yL*yL - 2.0)*sin(yL)       )*pow(args[2],-5)/6.0
	       + (yL*sin(yL) + 2.0*cos(yL) - 2.0)*pow(args[2],-4) -     args[0]*args[0]*(yL*(yL*yL - 18.0)*sin(yL) + 6.0*(yL*yL - 4.0)*cos(yL) + 24.0)*pow(args[2],-6)/6.0; 
    }
    else
      gain = 0.25*pow(L_v,4) + pow(L_v,6)*(args[0]*args[0] + args[2]*args[2])*(1.0/180.0 - 1.0/30.0);

    //LOSS
    double zL = L_v*args[1];

    if(args[1] > 1e-8) {
      origFlag2 = 1;
      loss = (cos(zL)*(2.0 - zL*zL) + 2.0*zL*sin(zL) - 2.0)*pow(args[1],-4);
    }
    else
      loss = 0.25*pow(L_v,4) - args[1]*args[1]*pow(L_v,6)/36.0;

    result = gain - loss;
  }*/
  /*
    if(fabs(gain - loss - result) > max) {
      max = fabs(gain-loss-result);
    
      printf("%g %le %le \n", max, result, gain-loss);
    } 
  */    
  return prefactor*result;
}

//this generates the convolution weights G
void generate_conv_weights_iso(double **conv_weights)
{
  int i, j, k, l, m, n;

#pragma omp parallel for private(i,j,k,l,m,n)
  for(i=0;i<N;i++)
  for(j=0;j<N;j++)
  for(k=0;k<N;k++) 
  for(l=0;l<N;l++)
  for(m=0;m<N;m++)
  for(n=0;n<N;n++) {

    conv_weights[k + N*(j + N*i)][n + N*(m + N*l)] = wtN[l]*wtN[m]*wtN[n]*0.25*pow(0.5*(diam_i+diam_j),2) * gHat3(zeta[l], zeta[m], zeta[n], zeta[i], zeta[j], zeta[k]);
    //conv_weights[k + N*(j + N*i)][n + N*(m + N*l)] = wtN[l]*wtN[m]*wtN[n]*gHat3(zeta[l], zeta[m], zeta[n], zeta[i], zeta[j], zeta[k]);
    //conv_weights[k + N*(j + N*i)][n + N*(m + N*l)] = 0.0;
  }
}

