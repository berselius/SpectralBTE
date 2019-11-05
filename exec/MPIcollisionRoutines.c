#include "MPIcollisionRoutines.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include "gauss_legendre.h"
#include <mpi.h>
#include <gsl/gsl_errno.h>

//kludgy, but whatever...
extern int GL, rank, numNodes, N;
extern double glance, lambda,Z, lambda_d, L_v, *eta;
extern double Gamma_couple;

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
double C_1 = (1.602e-38) / (4.0 * M_PI * 8.854e-12 * 9.109e-31 ); 

double ghat_theta(double theta, void* args) {
  struct integration_args intargs = *((struct integration_args *)args);
	// I_3,1 = \int_\sqrt(\theta_m)^\pi ghat_theta dth
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
double r = intargs.arg3;
  //eps-linear cross section
  // double bcos = eightPi*(glance/(theta*theta))*pow(theta,-2.0);
  //Rutherford xsec
  //double bcos = (cos(0.5*theta)/pow(sin(0.5*theta),3) ) / (-M_PI*log(sin(0.5*glance)));
  double bcos = pow(C_1, 2)*cos(0.5*theta)/(4*pow(sin(0.5*theta),3)); 
  return bcos*(cos(intargs.arg7*(1-cos(theta)) - intargs.arg6) * gsl_sf_bessel_J0(intargs.arg8*sin(theta)) - intargs.arg9);
}

//Computes the Taylor expansion portion
double ghat_theta2(double theta, void* args) {

  double *dargs = (double *)args;

  double r = dargs[0];
  double cosphi = dargs[1];
  double sinphi = dargs[2];
  double zetalen = dargs[3];
  double zetadot = dargs[4];

double theta_m = 2*atan(C_1/(pow(r,2)*lambda_d));
//double theta_m = 2;
  double c1 = 0.5*r*zetalen*cosphi;
  double c2 = 0.5*r*zetalen*sinphi;
  double c3 = r*zetadot*cosphi;

  return eightPi*( ((theta_m/theta)/theta)*(-0.25*c2*c2*cos(c3) + 0.5*c1*sin(c3)));
  //return (8.0/M_PI)*( ((glance/theta)/theta)*(-0.25*c2*c2*cos(c3) + 0.5*c1*sin(c3)) + (glance/192.0)*(-8.0*(3.0*c2*c2 +1)*c1*sin(c3) -24.0*c1*c1*cos(c3) + c2*c2*(3.0*c2 + 16.0)*cos(c3)));
}

double ghat_phi(double phi, void* args) {
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
 
 double theta_m = 2*atan(C_1/(pow(r,2)*lambda_d));
 //double theta_m = 2;

  intargs.arg4 = cos(phi);
  intargs.arg5 = sin(phi);
  intargs.arg6 = r * intargs.arg4 * intargs.arg1;
  intargs.arg7 = 0.5 * r * intargs.arg4 * intargs.arg0;
  intargs.arg8 = 0.5 * r * intargs.arg5 * intargs.arg0;
  intargs.arg9 = cos(r * intargs.arg4 * intargs.arg1);

  F_th.params = &intargs;

  double B = 0.5*r*intargs.arg0*intargs.arg4;
  double C = 0.5*r*intargs.arg0*intargs.arg5;
  double A = r*intargs.arg1*intargs.arg4;
  double Cons = (C*C*cos(A) + B*sin(A));

  if(theta_m > 10e-3)
   gsl_integration_cquad(&F_th ,theta_m ,M_PI,1e-6,1e-6,intargs.w_th,&result1,NULL,NULL); 
  else 
    result1 = Cons*(2.0/M_PI)*log(theta_m);

  //return intargs.arg5*gsl_sf_bessel_J0(intargs.arg3*intargs.arg5*intargs.arg2)*(result1 + result2);
 return intargs.arg5*gsl_sf_bessel_J0(intargs.arg3*intargs.arg5*intargs.arg2)*(result1 );
}

double ghat_r(double r, void* args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, error;

  gsl_function F_ph = intargs.F_ph;

  intargs.arg3 = r;

  F_ph.params = &intargs;

 // I_2 = \int_0^\pi F_ph dph
  int status;
  status = gsl_integration_qag(&F_ph,0,M_PI,1e-6,1e-6,6,1000,intargs.w_ph,&result,&error);
  if (status == GSL_EMAXITER) {
    printf("phi integration failed %g %g %g\n",intargs.arg0, intargs.arg1, intargs.arg2);
    exit(-1);
  }


  return pow(r,lambda+2)*result;
}

/*
function gHat3
--------------
computes integral for each convolution weight using gauss-legendre quadrature
inputs
ki, eta: wavenumbers for the convolution weight
 */

double gHat3(double zeta1, double zeta2, double zeta3, double xi1, double xi2, double xi3) {
  double result, error;
  //double args[3];
  gsl_integration_workspace *w_r  = gsl_integration_workspace_alloc(1000);
  gsl_function F_r, F_th, F_ph;
  F_r.function = &ghat_r;
  F_th.function = &ghat_theta;
  F_ph.function = &ghat_phi;

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


  //result = 4.0*M_PI*M_PI*gauss_legendre(GL, ghat_r, args, 0, L_v);
  F_r.params = &intargs;  

  int status;

  status = gsl_integration_qag(&F_r,0,L_v,1e-6,1e-6,6,1000,w_r,&result,&error);

  if (status == GSL_EMAXITER) {
    printf("r integration failed %g %g %g\n",zetalen,xizeta/zetalen,xiperp);
    exit(-1);
  }

  gsl_integration_workspace_free(w_r);
  gsl_integration_cquad_workspace_free(intargs.w_th);
  gsl_integration_workspace_free(intargs.w_ph);

  return 4*M_PI*M_PI*result;
}

double ghatL2(double theta, void* args) {
  double *dargs = (double *)args;
  double r = dargs[4];

  return sin(theta)*gsl_sf_bessel_J0(r*dargs[0]*sin(theta))*(-r*dargs[1]*sin(theta)*sin(theta)*cos(r*dargs[2]*cos(theta)) + 4*r*dargs[3]*sin(dargs[2]*cos(theta))*cos(theta));
}

double ghatL_couple(double r, void* args) {
  double *dargs = (double *)args;
  dargs[4] = r;

  return pow(r,lambda+3)*log(1 + pow(Gamma_couple,-3.0)*pow(r,4)) / log(1 + pow(Gamma_couple,-3.0)) *gauss_legendre(GL,ghatL2,dargs,0,M_PI);
}

double ghatL(double r, void* args) {
  double *dargs = (double *)args;
  dargs[4] = r;

  return pow(r,lambda+3) * gauss_legendre(GL,ghatL2,dargs,0,M_PI);
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

  if(Gamma_couple == 0)
    result = 2.0*M_PI*gauss_legendre(GL, ghatL, args, 0, L_v);
  else
    result = 2.0*M_PI*gauss_legendre(GL, ghatL_couple, args, 0, L_v);
    
  return result;
}

//this generates the convolution weights G_hat(zeta,xi)
void generate_conv_weights(double **conv_weights)
{
  int i, j, k, l, m, n, z;

  //double glancecons = 8.0*glance/M_PI;

 gsl_set_error_handler_off();

  //zeta iteration
  #pragma omp parallel for private(i,j,k,l,m,n,z)
  for(z=rank*(N*N*N/numNodes);z<(rank+1)*(N*N*N/numNodes);z++) {
    k = z % N;
    j = ((z-k)/N) % N;
    i = (z - k - N*j)/(N*N);
    printf("Rank: %d value: %d %d %d\n",rank,i,j,k);
    //xi iteration
    for(l=0;l<N;l++)
      for(m=0;m<N;m++) {
	for(n=0;n<N;n++) {
          if(glance == 0) {
            conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)] = gHat3L(eta[i], eta[j], eta[k],eta[l], eta[m], eta[n]);
	  }else{  
	    conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)] = gHat3(eta[i], eta[j], eta[k],eta[l], eta[m], eta[n]);}
	  if(isnan(conv_weights[z%(N*N*N/numNodes)][n + N*(m + N*l)]))
	    printf("%g %g %g %g %g %g\n",eta[i],eta[j],eta[k],eta[l],eta[m],eta[n]);
	}
      }
    
  }
}

