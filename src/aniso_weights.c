#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <gsl/gsl_integration.h>
#include <mpi.h>
#include "aniso_weights.h"

struct integration_args {
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //r cosphi 0.5 zetalen
  double arg8; //0.5 r zetalen sinphi
  double arg9; //cos(r cosphi zetadot)
  gsl_integration_cquad_workspace *w_th;
  gsl_function F_th;
  gsl_integration_workspace *w_ph;
  gsl_function F_ph;
};

const double eightPi = 8.0/M_PI;

static int N;
static double *eta;
static double L_v;
static int weight_flag;
static double glance;
static int numNodes;
static int rank;
static gsl_integration_glfixed_table *GL_table;
static double lambda;

void initialize_weights_AnIso(int nodes, double *zeta, double Lv, double lam, int weightFlag, double **conv_weights, double glance) {
  N = nodes;
  eta = zeta;
  L_v = Lv;
  weight_flag = weightFlag;
  lambda = lam;

  GL_table = gsl_integration_glfixed_table_alloc(64);

  FILE *fidWeights;
  char buffer_weights[100];
  char output_buffer[100];
  size_t readFlag;

  MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Status status;

  N = nodes;
  zeta = eta;
  L_v = Lv;
  
  int i,j;

  for(i=0;i<N*N*N;i++) {
    conv_weights[i] = malloc(N*N*N*sizeof(double));
  }

  if(glance == 0)
    sprintf(buffer_weights,"Weights/N%d_Aniso_L_v%g_lambda%g_Landau.wts",N,L_v,lambda);
  else
    sprintf(buffer_weights,"Weights/N%d_AnIso_L_v%g_lambda%g_glance%g.wts",N,L_v,lambda,glance);


  if(weightFlag == 0) { //Check to see if the weights are there
    if((fidWeights = fopen(buffer_weights,"r"))) {
      printf("Loading weights from file %s\n",buffer_weights);
      for(i=0;i<N*N*N;i++) { 
	int readFlag = (int)fread(conv_weights[i],sizeof(double),N*N*N,fidWeights);
	if(readFlag != N*N*N) {
	  printf("Error reading weight file\n");
	  exit(1);
	} 
      }      
    }
    else {
      printf("Stored weights not found for this configuration, generating ...\n");
      generate_conv_weights_AnIso(conv_weights);
      
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
  } 
  else { //weights forced to be regenerated
    printf("Fresh version of weights being computed and stored for this configuration\n");
    generate_conv_weights_AnIso(conv_weights);

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
  printf("Finished with weights\n");
}



double ghat_theta_AnIso(double theta, void* args) {
  struct integration_args intargs = *((struct integration_args *)args);

  //HARD SPHERES
  //gsl_integration_qag(&F_ph,0,M_PI,1e-2,1e-6,6,10000,w_ph,&result,&error);
  //return sin(theta)*(1.0/(4.0*M_PI))*result;

  //return sin(theta)*(1.0/(4.0*M_PI))*gauss_legendre(64,ghat_phi,dargs,0,M_PI);

  //printf("%g %g %g\n",intargs.arg0,intargs.arg1,intargs.arg2);

  /*
    Just to remind ourselves...
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //0.5 r cosphi zetalen
  double arg8; //0.5 r sinphi zetalen 
  double arg9; //cos(r cosphi zetadot)
  */

  //Linear convergence case
  //double bcos = eightPi*(glance/(theta*theta))*pow(theta,-2.0);
  //Coulomb case
  double bcos = (cos(0.5*theta)/pow(sin(0.5*theta),3) ) / (-M_PI*log(sin(0.5*glance)));

  return bcos*(cos(intargs.arg7*(1-cos(theta)) - intargs.arg6) * j0(intargs.arg8*sin(theta)) - intargs.arg9);
}

//Computes the Taylor expansion portion
double ghat_theta2(double theta, void* args) {

  double *dargs = (double *)args;

  double r = dargs[0];
  double cosphi = dargs[1];
  double sinphi = dargs[2];
  double zetalen = dargs[3];
  double zetadot = dargs[4];

  double c1 = 0.5*r*zetalen*cosphi;
  double c2 = 0.5*r*zetalen*sinphi;
  double c3 = r*zetadot*cosphi;

  return eightPi*( ((glance/theta)/theta)*(-0.25*c2*c2*cos(c3) + 0.5*c1*sin(c3)));
  //return (8.0/M_PI)*( ((glance/theta)/theta)*(-0.25*c2*c2*cos(c3) + 0.5*c1*sin(c3)) + (glance/192.0)*(-8.0*(3.0*c2*c2 +1)*c1*sin(c3) -24.0*c1*c1*cos(c3) + c2*c2*(3.0*c2 + 16.0)*cos(c3)));
}

double ghat_phi_AnIso(double phi, void* args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result1,result2;

  gsl_function F_th = intargs.F_th;
  
  double r = intargs.arg3;

  /*
    Just to remind ourselves...
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //0.5 r cosphi zetalen
  double arg8; //0.5 r sinphi zetalen 
  double arg9; //cos(r cosphi zetadot)
  */

  intargs.arg4 = cos(phi);
  intargs.arg5 = sin(phi);
  intargs.arg6 = r * intargs.arg4 * intargs.arg1;
  intargs.arg7 = 0.5 * r * intargs.arg4 * intargs.arg0;
  intargs.arg8 = 0.5 * r * intargs.arg5 * intargs.arg0;
  intargs.arg9 = cos(r * intargs.arg4 * intargs.arg1);

  F_th.params = &intargs;
  //F_th2.params = &thargs;
  
  gsl_integration_cquad(&F_th ,sqrt(glance),M_PI        ,1e-6,1e-6,intargs.w_th,&result1,NULL,NULL);  //"good" part
  //gsl_integration_qag(&F_th ,sqrt(glance),M_PI        ,1e-6,1e-6,6,10000,intargs.w_th,&result1,&error);  //"good" part
  //gsl_integration_cquad(&F_th2,glance      ,sqrt(glance),1e-6,1e-6,w_th,&result2,NULL,NULL);  //singular part

  //analytically solve the singular part with Taylor expansion
  double c1 = 0.5*r*intargs.arg0*intargs.arg4;
  double c2 = 0.5*r*intargs.arg0*intargs.arg5;
  double c3 = r*intargs.arg1*intargs.arg4;
  double C = (-0.25*c2*c2*cos(c3) + 0.5*c1*sin(c3));

  //Linear case
  //result2 = C*eightPi*(1 - sqrt(glance));
  //Coulomb case
  result2 = C/(2.0*M_PI);

  //gsl_integration_cquad_workspace_free(w_th);
  //gsl_integration_workspace_free(w_th);

  return intargs.arg5*j0(intargs.arg3*intargs.arg5*intargs.arg2)*(result1 + result2);
}


double ghat_r_AnIso(double r, void* args) {
  struct integration_args intargs = *((struct integration_args *)args);
  //double phargs[4];
  double result, error;

  //gsl_integration_workspace *w_ph = gsl_integration_workspace_alloc(1000);
  gsl_function F_ph = intargs.F_ph;

  /*
  phargs[0] = dargs[0]; //zetalen
  phargs[1] = dargs[1]; //xizeta/zetalen
  phargs[2] = dargs[2]; //xiperp
  phargs[3] = r;
  */

  intargs.arg3 = r;

  F_ph.params = &intargs;
  gsl_integration_qag(&F_ph,0,M_PI,1e-6,1e-6,6,1000,intargs.w_ph,&result,&error);

  //gsl_integration_workspace_free(w_ph);

  return pow(r,lambda+2)*result;
}

/*
function gHat3
--------------
computes integral for each convolution weight using gauss-legendre quadrature
inputs
ki, eta: wavenumbers for the convolution weight
 */

double gHat3_AnIso(double zeta1, double zeta2, double zeta3, double xi1, double xi2, double xi3) {
  double result, error;
  //double args[3];
  gsl_integration_workspace *w_r  = gsl_integration_workspace_alloc(1000);
  gsl_function F_r, F_th, F_ph;
  F_r.function = &ghat_r_AnIso;
  F_th.function = &ghat_theta_AnIso;
  F_ph.function = &ghat_phi_AnIso;

  /*
  w_th = gsl_integration_cquad_workspace_alloc(10000);
  
  F_th.function = &ghat_theta;
  F_th2.function = &ghat_theta2;
  
  w_ph = gsl_integration_workspace_alloc(10000);
  F_ph.function = &ghat_phi;
  */

  struct integration_args intargs;

  double zetalen2 = zeta1*zeta1 + zeta2*zeta2 + zeta3*zeta3;
  double xilen2   = xi1*xi1 + xi2*xi2 + xi3*xi3;
  double xizeta   = xi1*zeta1 + xi2*zeta2 + xi3*zeta3;
  double zetalen  = sqrt(zetalen2);
  double xiperp;


  if( ((xilen2 - xizeta*xizeta/zetalen2) < 0) || (zetalen2 == 0))
    xiperp = 0;
  else
    xiperp = sqrt( xilen2 - xizeta*xizeta/zetalen2);


  //args[0] = zetalen;
  //args[1] = xizeta/zetalen;
  //args[2] = xiperp;

  intargs.arg0 = zetalen;
  if(zetalen != 0)
    intargs.arg1 = xizeta/zetalen;
  else
    intargs.arg1 = 0.0;
  intargs.arg2 = xiperp;
  intargs.w_th = gsl_integration_cquad_workspace_alloc(1000);
  intargs.F_th = F_th;
  intargs.w_ph = gsl_integration_workspace_alloc(1000);
  intargs.F_ph = F_ph;

  //printf("%g %g %g\n",zetalen,xizeta/zetalen,xiperp);

  //result = 4.0*M_PI*M_PI*gauss_legendre(GL, ghat_r, args, 0, L_v);
  F_r.params = &intargs;  

  gsl_integration_qag(&F_r,0,L_v,1e-6,1e-6,6,1000,w_r,&result,&error);

  gsl_integration_workspace_free(w_r);
  gsl_integration_cquad_workspace_free(intargs.w_th);
  gsl_integration_workspace_free(intargs.w_ph);

  return 4*M_PI*M_PI*result;
}

double ghatL2(double theta, void* args) {
  double *dargs = (double *)args;
  double r = dargs[4];

  return sin(theta)*j0(r*dargs[0]*sin(theta))*(-r*r*dargs[1]*sin(theta)*sin(theta)*cos(r*dargs[2]*cos(theta)) + 4*r*dargs[3]*sin(r*dargs[2]*cos(theta))*cos(theta));
}

double ghatL(double r, void* args) {
  double *dargs = (double *)args;
  dargs[4] = r;

  gsl_function F_2;
  F_2.function = ghatL2;
  F_2.params = dargs;

  return pow(r,lambda+2)*gsl_integration_glfixed(&F_2,0,M_PI,GL_table);
}

double gHat3L(double zeta1, double zeta2, double zeta3, double xi1, double xi2, double xi3) {
  double result = 0.0;
  double args[5];

  double zetalen2 = zeta3*zeta3 + zeta2*zeta2 + zeta1*zeta1;
  double xilen2   = xi1*xi1     + xi2*xi2     + xi3*xi3;
  double xizeta   = xi1*zeta1   + xi2*zeta2   + xi3*zeta3;
  double zetalen  = sqrt(zetalen2);
  double xiperp;

  if( ((xilen2 - xizeta*xizeta/zetalen2) < 0) || (zetalen2 == 0))
    xiperp = 0;
  else
    xiperp = sqrt( xilen2 - xizeta*xizeta/zetalen2);


  args[0] = xiperp;
  args[1] = zetalen2;
  if(zetalen != 0)
    args[2] = xizeta/zetalen;
  else
    args[2] = 0.0;
  args[3] = zetalen;

  gsl_function F_ghat;
  F_ghat.function = ghatL;
  F_ghat.params = args;

  result = 2.0*M_PI*gsl_integration_glfixed(&F_ghat,0,L_v,GL_table);

  return result;
}

//this generates the convolution weights G_hat(zeta,xi)
void generate_conv_weights_AnIso(double **conv_weights) {
  int i, j, k, l, m, n, z;

  //zeta iteration
  #pragma omp parallel for private(i,j,k,l,m,n,z)
  for(z=rank*(N*N*N/numNodes);z<(rank+1)*(N*N*N/numNodes);z++) {
    k = z % N;
    j = ((z-k)/N) % N;
    i = (z - k - N*j)/(N*N);
    //xi iteration
    for(l=0;l<N;l++)
      for(m=0;m<N;m++) {
	for(n=0;n<N;n++) {
          if(glance == 0)
	    conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)] = gHat3L(eta[i], eta[j], eta[k],eta[l], eta[m], eta[n]);
	  else  
	    conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)] = gHat3_AnIso(eta[i], eta[j], eta[k],eta[l], eta[m], eta[n]);
	  //if(isnan(conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)]))
	  //printf("%g %g %g %g %g %g\n",eta[i],eta[j],eta[k],eta[l],eta[m],eta[n]);

	}
      }
    
  }
}

