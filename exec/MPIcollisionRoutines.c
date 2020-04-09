#include "MPIcollisionRoutines.h"
#include "gauss_legendre.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

// kludgy, but whatever...
extern int GL, rank, numNodes, N;
extern double glance, lambda, Z, lambda_d, L_v, *eta;
extern double Gamma_couple;

struct integration_args {
  double zetalen; // zetalen
  double xizeta_over_zetalen; // xizeta/zetalen
  double xiperp; // xiperp
  double r; // r
  double cosphi; // cosphi
  double sinphi; // sinphi
  double rcosphi_zetadotxi_over_zetalen; // r cosphi zetadotxi_over_zetalen
  double half_rcosphi_zetadotxi_over_zetalen; // r cosphi 0.5 zetalen
  double half_rsinphi_zetadotxi_over_zetalen; // 0.5 r zetalen sinphi
  double cos_rcosphi_zetadotxi_over_zetalen; // cos(r cosphi zetadotxi_over_zetalen)
  double zeta1;
  double zeta2;
  double zeta3;
  double xi1;
  double xi2;
  double xi3;

  gsl_integration_cquad_workspace *w_th;
  gsl_function F_th;
  gsl_integration_cquad_workspace *w_thE;
  gsl_function F_thE;
  gsl_integration_workspace *w_ph;
  gsl_function F_ph;
  gsl_integration_workspace *w_phE;
  gsl_function F_phE;
};

const double eightPi = 8.0 / M_PI;
double C_1 = (1.602e-38) / (8.0 * M_PI * 8.854e-12 * 9.109e-31);

double ghat_theta(double theta, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  // I_3,1 = \int_\sqrt(\theta_m)^\pi ghat_theta dth

  //double theta_m = 2 * atan(C_1 / (pow(r, 2) * lambda_d));
  // eps-linear cross section
  // double bcos = eightPi*(glance/(theta*theta))*pow(theta,-2.0);
  // Rutherford xsec
  // double bcos = (cos(0.5*theta)/pow(sin(0.5*theta),3) ) /
  // (-M_PI*log(sin(0.5*glance)));
  double bcos = cos(0.5 * theta) / (pow(sin(0.5 * theta), 3)); // / log(sin(0.5*theta_m));
  return bcos * (cos(intargs.half_rcosphi_zetadotxi_over_zetalen * (1 - cos(theta)) - intargs.rcosphi_zetadotxi_over_zetalen) *
                     gsl_sf_bessel_J0(intargs.half_rsinphi_zetadotxi_over_zetalen * sin(theta)) -
                 intargs.cos_rcosphi_zetadotxi_over_zetalen);
}

double ghat_thetaE(double theta, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  // I_3,1 = \int_\sqrt(\theta_m)^\pi ghat_theta dth

  double r = intargs.r;
  double B = 0.5 * intargs.zetalen *
             intargs.cosphi; // note: no 'r' included in this definition of B as
                           // it is factored out in the below expansion
  //double C = 0.5 * r * intargs.zetalen * intargs.sinphi;
  double A = r * intargs.xizeta_over_zetalen * intargs.cosphi;

  double bcos =  cos(0.5 * theta) / (sin(0.5 * theta)); // / log(sin(0.5*theta_m));
  return bcos *
         (-B * sin(A) - r * B * B * pow(sin(0.5 * theta), 2) * cos(A) +
          2.0 / 3.0 * r * r * B * B * B * pow(sin(0.5 * theta), 4) * sin(A));
}

// Computes the Taylor expansion portion
double ghat_theta2(double theta, void *args) {

  double *dargs = (double *)args;

  double r = dargs[0];
  double cosphi = dargs[1];
  double sinphi = dargs[2];
  double zetalen = dargs[3];
  double zetadotxi_over_zetalen = dargs[4];

  //double u = 4.11e5;
  double theta_m = 2 * atan(2*C_1 / (pow(r, 2) * lambda_d));
  //double theta_m = 1e-9;
// double theta_m = 10e-5;
  // printf("theta_m = %g \n", theta_m);

  double c1 = 0.5 * r * zetalen * cosphi;
  double c2 = 0.5 * r * zetalen * sinphi;
  double c3 = r * zetadotxi_over_zetalen * cosphi;

  return eightPi * (((theta_m / theta) / theta) *
                    (-0.25 * c2 * c2 * cos(c3) + 0.5 * c1 * sin(c3)));
  // return (8.0/M_PI)*( ((glance/theta)/theta)*(-0.25*c2*c2*cos(c3) +
  // 0.5*c1*sin(c3)) + (glance/192.0)*(-8.0*(3.0*c2*c2 +1)*c1*sin(c3)
  // -24.0*c1*c1*cos(c3) + c2*c2*(3.0*c2 + 16.0)*cos(c3)));
}

double ghat_phi(double phi, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, result1, result2;

  gsl_function F_th = intargs.F_th;

  int status;

  double r = intargs.r;
  //double u = 4.11e5;
  double theta_m = 2 * atan(2*C_1 / (pow(r, 2) * lambda_d));
  //double theta_m = 1e-9;
  intargs.cosphi = cos(phi);
  intargs.sinphi = sin(phi);
  intargs.rcosphi_zetadotxi_over_zetalen = r * intargs.cosphi * intargs.xizeta_over_zetalen;
  intargs.half_rcosphi_zetadotxi_over_zetalen = 0.5 * r * intargs.cosphi * intargs.zetalen;
  intargs.half_rsinphi_zetadotxi_over_zetalen = 0.5 * r * intargs.sinphi * intargs.zetalen;
  intargs.cos_rcosphi_zetadotxi_over_zetalen = cos(r * intargs.cosphi * intargs.xizeta_over_zetalen);

  F_th.params = &intargs;

  double B = 0.5 * r * intargs.zetalen * intargs.cosphi;
  double C = 0.5 * r * intargs.zetalen * intargs.sinphi;
  double A = r * intargs.xizeta_over_zetalen * intargs.cosphi;

  if (theta_m > 1e-4) {
    status = gsl_integration_cquad(&F_th, theta_m, M_PI, 1e-6, 1e-6, intargs.w_th,
                          &result, NULL, NULL);
  } // computes full integral, gets stored in "result"
  else {
    status = gsl_integration_cquad(&F_th, sqrt(theta_m), M_PI, 1e-6, 1e-6, intargs.w_th,
                          &result1, NULL, NULL); // stored in "result1"
    result2 = (0.5*C*C*cos(A) - B*sin(A))* log(theta_m);   // 
    result = result1 + result2;
  }

  if(status) {
    if(status == GSL_EMAXITER)
      printf("Max iterations reached\n");
    else if(status == GSL_EROUND)
      printf("Roundoff error detected\n");
    else if(status == GSL_ESING)
      printf("Nonintegrable singularity detected\n");
    else if(status == GSL_EDIVERGE)
      printf("Integral appears to be divergent\n");
    else if(status == GSL_EDOM)
      printf("Input argument error\n");
    exit(37);
  }

  // printf("taylor expansion computed at theta_m = %g and yielded a result2= %g
  // \n", theta_m, result2);}   //add taylor expansion for small theta_m values

  // return
  // intargs.sinphi*gsl_sf_bessel_J0(intargs.r*intargs.sinphi*intargs.xiperp)*(result1
  // + result2);
  return intargs.sinphi *
         gsl_sf_bessel_J0(intargs.r * intargs.sinphi * intargs.xiperp) *
         (result);
}

double ghat_phiE(double phi, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, result1, result2;

  gsl_function F_thE = intargs.F_thE;

  double r = intargs.r;
 // double u = 4.11e5;
  double theta_m = 2 * atan(2*C_1 / (pow(r, 2) * lambda_d));
 // double theta_m = 1e-9;
  intargs.cosphi = cos(phi);
  intargs.sinphi = sin(phi);
  intargs.rcosphi_zetadotxi_over_zetalen = r * intargs.cosphi * intargs.xizeta_over_zetalen;
  intargs.half_rcosphi_zetadotxi_over_zetalen = 0.5 * r * intargs.cosphi * intargs.zetalen;
  intargs.half_rsinphi_zetadotxi_over_zetalen = 0.5 * r * intargs.sinphi * intargs.zetalen;
  intargs.cos_rcosphi_zetadotxi_over_zetalen = cos(r * intargs.cosphi * intargs.xizeta_over_zetalen);

  F_thE.params = &intargs;

  double B = 0.5 * r * intargs.zetalen * intargs.cosphi;
  double C = 0.5 * r * intargs.zetalen * intargs.sinphi;
  double A = r * intargs.xizeta_over_zetalen * intargs.cosphi;

  if (theta_m > 1e-4) {
    gsl_integration_cquad(&F_thE, theta_m, M_PI, 1e-6, 1e-6, intargs.w_thE,
                          &result, NULL, NULL);
  } //"good" part gets stored in "result"
  else {
    gsl_integration_cquad(&F_thE, sqrt(theta_m), M_PI, 1e-6, 1e-6,
                          intargs.w_thE, &result1, NULL,
                          NULL);               // stored in "result1"
    result2 = (0.5*C*C*cos(A) - B*sin(A)) * log(theta_m); // 
    result = result1 + result2;
  }

  return intargs.sinphi *
         gsl_sf_bessel_J0(intargs.r * intargs.sinphi * intargs.xiperp) *
         (result);
}

double ghat_r(double r, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, error;

  gsl_function F_ph = intargs.F_ph;

  intargs.r = r;

  F_ph.params = &intargs;

  // I_2 = \int_0^\pi F_ph dph
  int status;
  status = gsl_integration_qag(&F_ph, 0, M_PI, 1e-6, 1e-6, 10000, 6, intargs.w_ph,
                                &result, &error);
  
  if(status) {
    if (status == GSL_EMAXITER) {
      printf("phi integration failed %g %g %g %g %g %g %g %g %g %g result %g error %g\n",
	     intargs.r, intargs.zeta1, intargs.zeta2, intargs.zeta3,
	     intargs.zetalen, intargs.xi1, intargs.xi2, intargs.xi3,
	     intargs.xizeta_over_zetalen, intargs.xiperp, result, error);
    }
    else {
      printf("Some other error occured in phi integration\n");
    }
    exit(37);
  }

  return pow(r, lambda + 2) * result;
}

double ghat_rE(double r, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, error;

  gsl_function F_phE = intargs.F_phE;

  intargs.r = r;

  F_phE.params = &intargs;

  // I_2 = \int_0^\pi F_ph dph
  int status;
  status = gsl_integration_qag(&F_phE, 0, M_PI, 1e-6, 1e-6, 10000, 6, intargs.w_phE,
                                &result, &error);
  if (status == GSL_EMAXITER) {
    printf("(expansion)phi integration failed %g %g %g %g %g %g %g %g %g %g\n",
           intargs.r, intargs.zeta1, intargs.zeta2, intargs.zeta3,
           intargs.zetalen, intargs.xi1, intargs.xi2, intargs.xi3,
           intargs.xizeta_over_zetalen, intargs.xiperp);
    exit(-1);
  }

  return result; // power of r cancels out
}

/*
function gHat3
--------------
computes integral for each convolution weight using gauss-legendre quadrature
inputs
ki, eta: wavenumbers for the convolution weight
 */

double gHat3(double zeta1, double zeta2, double zeta3, double xi1, double xi2,
             double xi3) {
  double result, result1, result2;
  // double args[3];
  // gsl_integration_workspace *w_r  = gsl_integration_workspace_alloc(10000);
  gsl_integration_cquad_workspace *w_r =
      gsl_integration_cquad_workspace_alloc(10000);
  gsl_integration_cquad_workspace *w_rE =
      gsl_integration_cquad_workspace_alloc(10000);

  gsl_function F_r, F_rE, F_th, F_thE, F_ph, F_phE;
  F_r.function = &ghat_r;
  F_rE.function = &ghat_rE;
  F_th.function = &ghat_theta;
  F_thE.function = &ghat_thetaE;
  F_ph.function = &ghat_phi;
  F_phE.function = &ghat_phiE;

  struct integration_args intargs;

  double zetalen2 = zeta1 * zeta1 + zeta2 * zeta2 + zeta3 * zeta3;
  double xilen2 = xi1 * xi1 + xi2 * xi2 + xi3 * xi3;
  double xizeta = xi1 * zeta1 + xi2 * zeta2 + xi3 * zeta3;
  double zetalen = sqrt(zetalen2);
  double xiperp;

  if (((xilen2 - xizeta * xizeta / zetalen2) < 0) || (zetalen2 == 0))
    xiperp = 0;
  else
    xiperp = sqrt(xilen2 - xizeta * xizeta / zetalen2);

  intargs.zetalen = zetalen;
  if (zetalen != 0)
    intargs.xizeta_over_zetalen = xizeta / zetalen;
  else
    intargs.xizeta_over_zetalen = 0.0;
  intargs.xiperp = xiperp;

  intargs.zeta1 = zeta1;
  intargs.zeta2 = zeta2;
  intargs.zeta3 = zeta3;
  intargs.xi1 = xi1;
  intargs.xi2 = xi2;
  intargs.xi3 = xi3;

  intargs.w_th = gsl_integration_cquad_workspace_alloc(1000);
  intargs.F_th = F_th;
  intargs.w_thE = gsl_integration_cquad_workspace_alloc(1000);
  intargs.F_thE = F_thE;
  intargs.w_ph = gsl_integration_workspace_alloc(10000);
  intargs.F_ph = F_ph;
  intargs.w_phE = gsl_integration_workspace_alloc(10000);
  intargs.F_phE = F_phE;

  F_r.params = &intargs;
  F_rE.params = &intargs;

  int status, status1;

  status = gsl_integration_cquad(&F_rE, 0, 1e-3, 1e-6, 1e-6, w_rE, &result1,
                                 NULL, NULL);
  status1 = gsl_integration_cquad(&F_r, 1e-3, L_v, 1e-6, 1e-6, w_r, &result2,
                                  NULL, NULL);

  if (status == GSL_EMAXITER) {
    printf("(expansion)r integration failed %g %g %g %g %g %g %g %g %g \n",
           zeta1, zeta2, zeta3, xi1, xi2, xi3, zetalen, xizeta / zetalen,
           xiperp);
    exit(-1);
  }
  if (status1 == GSL_EMAXITER) {
    printf("r integration failed %g %g %g %g %g %g %g %g %g \n", zeta1, zeta2,
           zeta3, xi1, xi2, xi3, zetalen, xizeta / zetalen, xiperp);
    exit(-1);
  }
  result = pow(C_1, 2)*(result1 + result2);

  // gsl_integration_workspace_free(w_r);
  gsl_integration_cquad_workspace_free(w_r);
  gsl_integration_cquad_workspace_free(w_rE);
  gsl_integration_cquad_workspace_free(intargs.w_th);
  gsl_integration_cquad_workspace_free(intargs.w_thE);
  gsl_integration_workspace_free(intargs.w_ph);
  gsl_integration_workspace_free(intargs.w_phE);

  return 2.0 * sqrt(2 * M_PI)  * result;
}

double ghatL2(double theta, void *args) {
  double *dargs = (double *)args;
  double r = dargs[4];

  return sin(theta) * gsl_sf_bessel_J0(r * dargs[0] * sin(theta)) *
         (-r*r * dargs[1] * sin(theta) * sin(theta) *
              cos(r * dargs[2] * cos(theta)) +
          4 * r * dargs[3] * sin(dargs[2] * cos(theta)) * cos(theta));
}

double ghatL_couple(double r, void *args) {
  double *dargs = (double *)args;
  dargs[4] = r;

  return pow(r, lambda + 3) * log(1 + pow(Gamma_couple, -3.0) * pow(r, 4)) /
         log(1 + pow(Gamma_couple, -3.0)) *
         gauss_legendre(GL, ghatL2, dargs, 0, M_PI);
}

double ghatL(double r, void *args) {
  double *dargs = (double *)args;
  dargs[4] = r;

double u = 1.69e11; 
double gamma = M_PI*C_1*C_1;
double C_L = 0.5*gamma*log(1 + (lambda_d*lambda_d*pow(u,4))/(4*C_1*C_1) );	
  

  
  return pow(r, lambda + 2) * C_L * gauss_legendre(GL, ghatL2, dargs, 0, M_PI);
}

double gHat3L(double zeta1, double zeta2, double zeta3, double xi1, double xi2,
              double xi3) {
  double result = 0.0;
  double args[5];

  double zetalen2 = zeta3 * zeta3 + zeta2 * zeta2 + zeta1 * zeta1;
  double xilen2 = xi1 * xi1 + xi2 * xi2 + xi3 * xi3;
  double xizeta = xi1 * zeta1 + xi2 * zeta2 + xi3 * zeta3;
  double zetalen = sqrt(zetalen2);
  double xiperp;

  if (((xilen2 - xizeta * xizeta / zetalen2) < 0) || (zetalen2 == 0))
    xiperp = 0;
  else
    xiperp = sqrt(xilen2 - xizeta * xizeta / zetalen2);

  args[0] = xiperp;
  args[1] = zetalen2;
  if (zetalen != 0)
    args[2] = xizeta / zetalen;
  else
    args[2] = 0.0;
  args[3] = zetalen;

  if (Gamma_couple == 0)
    result = 1.0/sqrt(2.0 * M_PI) * gauss_legendre(GL, ghatL, args, 0, L_v);
  else
    result = 1.0/sqrt(2.0 * M_PI) * gauss_legendre(GL, ghatL_couple, args, 0, L_v);

  return result;
}

// this generates the convolution weights G_hat(zeta,xi)
void generate_conv_weights(double **conv_weights) {
  int i, j, k, l, m, n, z;

  // double glancecons = 8.0*glance/M_PI;

  gsl_set_error_handler_off();

// zeta iteration
#pragma omp parallel for private(i, j, k, l, m, n, z)
  for (z = rank * (N * N * N / numNodes);
       z < (rank + 1) * (N * N * N / numNodes); z++) {
    k = z % N;
    j = ((z - k) / N) % N;
    i = (z - k - N * j) / (N * N);
    printf("Rank: %d value: %d %d %d\n", rank, i, j, k);
    // xi iteration
    for (l = 0; l < N; l++)
      for (m = 0; m < N; m++) {
        for (n = 0; n < N; n++) {
          if (glance == 0) {
            conv_weights[z % (N * N * N / numNodes)][n + N * (m + N * l)] =
                gHat3L(eta[i], eta[j], eta[k], eta[l], eta[m], eta[n]);
          } else {
            conv_weights[z % (N * N * N / numNodes)][n + N * (m + N * l)] =
                gHat3(eta[i], eta[j], eta[k], eta[l], eta[m], eta[n]);
          }
          if (isnan(conv_weights[z % (N * N * N / numNodes)]
                                [n + N * (m + N * l)]))
            printf("%g %g %g %g %g %g\n", eta[i], eta[j], eta[k], eta[l],
                   eta[m], eta[n]);
        }
      }
  }
}
