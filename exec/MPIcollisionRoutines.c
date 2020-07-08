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

//Thermal speed, used for calculating constant CL
static double v_th = 4.084e5;

struct integration_args {
  double zetalen;                             // zetalen
  double xizeta_over_zetalen;                 // xizeta/zetalen
  double xiperp;                              // xiperp
  double r;                                   // r
  double cosphi;                              // cosphi
  double sinphi;                              // sinphi
  double rcosphi_zetadotxi_over_zetalen;      // r cosphi zetadotxi_over_zetalen
  double half_rcosphi_zetadotxi_over_zetalen; // r cosphi 0.5 zetalen
  double half_rsinphi_zetadotxi_over_zetalen; // 0.5 r zetalen sinphi
  double cos_rcosphi_zetadotxi_over_zetalen;  // cos(r cosphi
                                              // zetadotxi_over_zetalen)
  double zeta1;
  double zeta2;
  double zeta3;
  double xi1;
  double xi2;
  double xi3;

  gsl_integration_cquad_workspace *w_th;
  gsl_function F_th;
  gsl_integration_cquad_workspace *w_th_small;
  gsl_function F_th_small;
  gsl_integration_workspace *w_ph;
  gsl_function F_ph;
  gsl_integration_workspace *w_ph_small;
  gsl_function F_ph_small;
};

const double eightPi = 8.0 / M_PI;
double C_1 = (2.566e-38) / (8.0 * M_PI * 8.854e-12 * 9.109e-31);

double I_three_Boltz(double theta, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  // I_3,1 = \int_\sqrt(\theta_m)^\pi ghat_theta dth
  // the "normal"/"non Taylor" part of I_3,1 and I_3,2

  // double theta_m = 2 * atan(C_1 / (pow(r, 2) * lambda_d));
  // eps-linear cross section
  // double bcos = eightPi*(glance/(theta*theta))*pow(theta,-2.0);
  // Rutherford xsec
  // double bcos = (cos(0.5*theta)/pow(sin(0.5*theta),3) ) /
  // (-M_PI*log(sin(0.5*glance)));
  // reminder:
  // A = intargs.rcosphi_zetadotxi_over_zetalen
  // B = intargs.half_rcosphi_zetadotxi_over_zetalen
  // C = intargs.half_rsinphi_zetadotxi_over_zetalen

  double bcos =
      cos(0.5 * theta) / (pow(sin(0.5 * theta), 3)); // / log(sin(0.5*theta_m));
  return bcos *
         (cos(intargs.half_rcosphi_zetadotxi_over_zetalen * (1 - cos(theta)) -
              intargs.rcosphi_zetadotxi_over_zetalen) *
              gsl_sf_bessel_J0(intargs.half_rsinphi_zetadotxi_over_zetalen *
                               sin(theta)) -
          intargs.cos_rcosphi_zetadotxi_over_zetalen);
  // bcos*(cos(B(1-cos(\theta))-A)*J_0(C*sin(\theta)) - cos(A))
  // this is stored in F_h
}

double I_three_Boltz_small(double theta, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);

  // I_3,3 integrand as in manuscript
  //\frac{\cos (\theta /2)}{ \sin ^{3}(\theta /2)}(\tilde{A}\tilde{B} -
  // \frac12\tilde{A}^2 - \frac14 \tilde{C}^2 )

  double tA =
      intargs.zetalen * intargs.cosphi * pow(sin(0.5 * theta), 2); // \tilde{A}
  double tB = intargs.xizeta_over_zetalen * intargs.cosphi;        // \tilde{B}
  double tC = 0.5 * intargs.zetalen * intargs.sinphi * sin(theta); // \tilde{C}

  double bcos = cos(0.5 * theta) / pow(sin(0.5 * theta), 3);
  return bcos * (tA * tB - 0.5 * tA * tA - 0.25 * tC * tC);
  // stored in F_th_small
}

// Computes the Taylor expansion portion
/*
double I_three_Boltz_small(double theta, void *args) {

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
*/

double I_two_Boltz_small(double phi, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, result1, result2;

  gsl_function F_th_small = intargs.F_th_small;

  double r = intargs.r;
  double u;
     
    //Uncomment this line if using constant cutoff
    //u = v_th;

    //Uncomment this line if using velocity dependent cutoff
    u = r;

  double theta_m = 2.0 * atan(2.0 * C_1 / (pow(u, 2) * lambda_d));
  // double theta_m = 1e-9;
  intargs.cosphi = cos(phi);
  intargs.sinphi = sin(phi);
  intargs.rcosphi_zetadotxi_over_zetalen =
      r * intargs.cosphi * intargs.xizeta_over_zetalen; // A
  intargs.half_rcosphi_zetadotxi_over_zetalen =
      0.5 * r * intargs.cosphi * intargs.zetalen; // B
  intargs.half_rsinphi_zetadotxi_over_zetalen =
      0.5 * r * intargs.sinphi * intargs.zetalen; // C
  intargs.cos_rcosphi_zetadotxi_over_zetalen =
      cos(r * intargs.cosphi * intargs.xizeta_over_zetalen); // cos(A)

  F_th_small.params = &intargs;

  double B = 0.5 * r * intargs.zetalen * intargs.cosphi;
  double C = 0.5 * r * intargs.zetalen * intargs.sinphi;
  double A = r * intargs.xizeta_over_zetalen * intargs.cosphi;

  if (theta_m > 1e-4) {
    gsl_integration_cquad(&F_th_small, theta_m, M_PI, 1e-6, 1e-6,
                          intargs.w_th_small, &result, NULL, NULL);
  } //"good" part gets stored in "result"
  else {
    gsl_integration_cquad(&F_th_small, sqrt(theta_m), M_PI, 1e-6, 1e-6,
                          intargs.w_th_small, &result1, NULL,
                          NULL); // stored in "result1"
    result2 = (0.5 * C * C * cos(A) - B * sin(A)) * log(theta_m); //
    result = result1 + result2;
  }

  return intargs.sinphi *
         gsl_sf_bessel_J0(intargs.r * intargs.sinphi * intargs.xiperp) *
         (result);
  // gets stored in F_ph_small
}

double I_two_Boltz(double phi, void *args) {
  // Computes the integrand of I_2 from the manuscript
  struct integration_args intargs = *((struct integration_args *)args);
  double result, result1, result2;

  gsl_function F_th = intargs.F_th;

  int status;

  double r = intargs.r;
  double u;
  //Uncomment this line if using constant cutoff 
  //u = v_th;

  //Uncomment this line if using velocity dependent cutoff
  u = r;
  
  double theta_m = 2.0 * atan(2.0 * C_1 / (pow(u, 2) * lambda_d));
  // double theta_m = 1e-9;
  intargs.cosphi = cos(phi);
  intargs.sinphi = sin(phi);
  intargs.rcosphi_zetadotxi_over_zetalen =
      r * intargs.cosphi * intargs.xizeta_over_zetalen; // A
  intargs.half_rcosphi_zetadotxi_over_zetalen =
      0.5 * r * intargs.cosphi * intargs.zetalen; // B
  intargs.half_rsinphi_zetadotxi_over_zetalen =
      0.5 * r * intargs.sinphi * intargs.zetalen; // C
  intargs.cos_rcosphi_zetadotxi_over_zetalen =
      cos(r * intargs.cosphi * intargs.xizeta_over_zetalen); // cos(A)

  F_th.params = &intargs;

  double B = 0.5 * r * intargs.zetalen * intargs.cosphi;
  double C = 0.5 * r * intargs.zetalen * intargs.sinphi;
  double A = r * intargs.xizeta_over_zetalen * intargs.cosphi;

  if (theta_m > 1e-4) {
    // computes full integral, gets stored in "result"
    // I_{3,1} + I_{3,2} when theta > epsilon_theta
    status = gsl_integration_cquad(&F_th, theta_m, M_PI, 1e-6, 1e-6,
                                   intargs.w_th, &result, NULL, NULL);
  } 
  else {
    // computes I_{3,1} + I_{3,2} when theta < epsilon_theta
    status = gsl_integration_cquad(&F_th, sqrt(theta_m), M_PI, 1e-6, 1e-6,
                                   intargs.w_th, &result1, NULL,
                                   NULL); // stored in "result1"
    result2 = 2.0 * (0.5 * C * C * cos(A) - B * sin(A)) * log(theta_m); //
    result = result1 + result2;
  }

  if (status) {
    if (status == GSL_EMAXITER)
      printf("Max iterations reached\n");
    else if (status == GSL_EROUND)
      printf("Roundoff error detected\n");
    else if (status == GSL_ESING)
      printf("Nonintegrable singularity detected\n");
    else if (status == GSL_EDIVERGE)
      printf("Integral appears to be divergent\n");
    else if (status == GSL_EDOM)
      printf("Input argument error\n");
    exit(37);
  }

  // printf("taylor expansion computed at theta_m = %g and yielded a result2= %g
  // \n", theta_m, result2);}   //add taylor expansiJon for small theta_m values

  return intargs.sinphi *
         gsl_sf_bessel_J0(intargs.r * intargs.sinphi * intargs.xiperp) *
         result;

  // sin\phi J_0(r * sin\phi |xi perp|) I_3
  // gets stored in F_ph
}

double I_one_Boltz(double r, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, error;

  gsl_function F_ph = intargs.F_ph;

  intargs.r = r;

  F_ph.params = &intargs;

  // I_2 = \int_0^\pi F_ph dph
  int status;
  status = gsl_integration_qag(&F_ph, 0, M_PI, 1e-6, 1e-6, 10000, 6,
                               intargs.w_ph, &result, &error);

  if (status) {
    if (status == GSL_EMAXITER) {
      printf("phi integration failed %g %g %g %g %g %g %g %g %g %g result %g "
             "error %g\n",
             intargs.r, intargs.zeta1, intargs.zeta2, intargs.zeta3,
             intargs.zetalen, intargs.xi1, intargs.xi2, intargs.xi3,
             intargs.xizeta_over_zetalen, intargs.xiperp, result, error);
    } else {
      printf("Some other error occured in phi integration\n");
    }
    exit(37);
  }

  return pow(r, lambda + 2) * result;
  // gets stored in F_r
}

double I_one_Boltz_small(double r, void *args) {
  struct integration_args intargs = *((struct integration_args *)args);
  double result, error;

  gsl_function F_ph_small = intargs.F_ph_small;

  intargs.r = r;

  F_ph_small.params = &intargs;

  // I_2 = \int_0^\pi F_ph dph
  int status;
  status = gsl_integration_qag(&F_ph_small, 0, M_PI, 1e-6, 1e-6, 10000, 6,
                               intargs.w_ph_small, &result, &error);
  if (status == GSL_EMAXITER) {
    printf("(expansion)phi integration failed %g %g %g %g %g %g %g %g %g %g\n",
           intargs.r, intargs.zeta1, intargs.zeta2, intargs.zeta3,
           intargs.zetalen, intargs.xi1, intargs.xi2, intargs.xi3,
           intargs.xizeta_over_zetalen, intargs.xiperp);
    exit(-1);
  }

  // We pick up two extra powers of r due to the expansion
  return pow(r, lambda + 4) * result;
}

/*
function gHat3_Boltz
--------------
computes integral for each convolution weight using gauss-legendre quadrature
inputs
ki, eta: wavenumbers for the convolution weight
 */

double gHat3_Boltz(double zeta1, double zeta2, double zeta3, double xi1,
                   double xi2, double xi3) {

  double result, result1, result2;

  gsl_integration_cquad_workspace *w_r =
      gsl_integration_cquad_workspace_alloc(10000);
  gsl_integration_cquad_workspace *w_r_small =
      gsl_integration_cquad_workspace_alloc(10000);

  gsl_function F_r, F_r_small, F_th, F_th_small, F_ph, F_ph_small;
  F_r.function = &I_one_Boltz;
  F_r_small.function = &I_one_Boltz_small;
  F_ph.function = &I_two_Boltz;
  F_ph_small.function = &I_two_Boltz_small;
  F_th.function = &I_three_Boltz;
  F_th_small.function = &I_three_Boltz_small;

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
  intargs.w_th_small = gsl_integration_cquad_workspace_alloc(1000);
  intargs.F_th_small = F_th_small;
  intargs.w_ph = gsl_integration_workspace_alloc(10000);
  intargs.F_ph = F_ph;
  intargs.w_ph_small = gsl_integration_workspace_alloc(10000);
  intargs.F_ph_small = F_ph_small;

  F_r.params = &intargs;
  F_r_small.params = &intargs;

  int status, status1;
  // split up the integration into two parts:
  status = gsl_integration_cquad(&F_r_small, 0, 1e-3, 1e-6, 1e-6, w_r_small,
                                 &result1, NULL, NULL); // small values of r
  status1 = gsl_integration_cquad(&F_r, 1e-3, L_v, 1e-6, 1e-6, w_r, &result2,
                                  NULL, NULL); // large values of r

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

  // add the two pieces together and multiply by outside constant
  // 2C_1^1\sqrt(2\pi) I_1
  result = 2.0 * pow(C_1, 2) * sqrt(2.0 * M_PI) * (result1 + result2);

  gsl_integration_cquad_workspace_free(w_r);
  gsl_integration_cquad_workspace_free(w_r_small);
  gsl_integration_cquad_workspace_free(intargs.w_th);
  gsl_integration_cquad_workspace_free(intargs.w_th_small);
  gsl_integration_workspace_free(intargs.w_ph);
  gsl_integration_workspace_free(intargs.w_ph_small);

  return result;
}

double I_two_Landau(double theta, void *args) {
  double *dargs = (double *)args;
  double r = dargs[4];

  return sin(theta) * gsl_sf_bessel_J0(r * dargs[0] * sin(theta)) *
         (-r * dargs[1] * sin(theta) * sin(theta) *
              cos(r * dargs[2] * cos(theta)) +
          4 * dargs[3] * sin(r * dargs[2] * cos(theta)) * cos(theta));
  // \sin\theta J_0(r\sin\theta |\m^\perp|) [4r|\k|\cos\theta \sin( r\cos\theta
  // \frac{\k \cdot \m}{|\k|}) - r^2 \sin^2\theta|\k|^2 \cos( r\cos\theta
  // \frac{\k \cdot \m}{|\k|})]
}

double I_one_Landau(double r, void *args) {
  double *dargs = (double *)args;
  dargs[4] = r;

  double u;

  //Set to this if using thermal speed cutoff
  //u = v_th;
  //Set to this if using velo-dep CL
  u = r;

  double C_L =
    0.5 * log(1 + pow(lambda_d * pow(u, 2) / (2.0 * C_1), 2));

  //Note: pulled out extra r from the Landau I2 formula
  return pow(r, lambda + 3) * C_L *
         gauss_legendre(GL, I_two_Landau, dargs, 0, M_PI);
}

double gHat3_Landau(double zeta1, double zeta2, double zeta3, double xi1,
                    double xi2, double xi3) {
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

  result = 1.0 / sqrt(2.0 * M_PI) * 2.0 * M_PI * C_1 * C_1 *
           gauss_legendre(GL, I_one_Landau, args, 0, L_v);

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
                gHat3_Landau(eta[i], eta[j], eta[k], eta[l], eta[m], eta[n]);
          } else {
            conv_weights[z % (N * N * N / numNodes)][n + N * (m + N * l)] =
                gHat3_Boltz(eta[i], eta[j], eta[k], eta[l], eta[m], eta[n]);
          }
          if (isnan(conv_weights[z % (N * N * N / numNodes)]
                                [n + N * (m + N * l)]))
            printf("%g %g %g %g %g %g\n", eta[i], eta[j], eta[k], eta[l],
                   eta[m], eta[n]);
        }
      }
  }
}
