#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <omp.h>
#include <stdarg.h>

#include "constants.h"
#include "collisions.h"
#include "conserve.h"
#include "momentRoutines.h"
#include "weights.h"

static fftw_plan p_forward;
static fftw_plan p_backward;
static fftw_complex *temp;
static fftw_complex *fftIn_f, *fftOut_f, *fftIn_g, *fftOut_g, *qHat;
static double *M_i, *M_j, *g_i, *g_j;
static double L_v;
static double L_eta;
static double *v;
static double *eta;
static double dv;
static double deta;
static int N;
static double *wtN;
static double scale3;

static int inverse = 1;
static int noinverse = 0;

static void find_maxwellians(double *M_mat, double *g_mat, double *mat);
static void compute_Qhat(double *f_mat, double *g_mat, int weightgenFlag, ...);

//Initializes this module's static variables and allocates what needs allocating
void initialize_coll(int nodes, double length, double *vel, double *zeta) {
  int i;

  N = nodes;
  L_v = length;
  v = vel;
  dv = v[1] - v[0];

  eta = zeta;
  deta = zeta[1]-zeta[0];;
  L_eta = -zeta[0];
  //L_eta = 0.0;

  scale3 = pow(1.0/sqrt(2.0*M_PI), 3.0);

  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  #pragma omp simd
  for(i=1;i<(N-1);i++) {
    wtN[i] = 1.0;
  }
  wtN[N-1] = 0.5;

  //SETTING UP FFTW

  //allocate bins for ffts
  fftIn_f = fftw_malloc(N*N*N*sizeof(fftw_complex));
  fftOut_f = fftw_malloc(N*N*N*sizeof(fftw_complex));
  fftIn_g = fftw_malloc(N*N*N*sizeof(fftw_complex));
  fftOut_g = fftw_malloc(N*N*N*sizeof(fftw_complex));
  qHat = fftw_malloc(N*N*N*sizeof(fftw_complex));
  temp = fftw_malloc(N*N*N*sizeof(fftw_complex));

  //Set up plans for FFTs
  p_forward  = fftw_plan_dft_3d (N, N, N, temp, temp, FFTW_FORWARD , FFTW_ESTIMATE);
  p_backward = fftw_plan_dft_3d (N, N, N, temp, temp, FFTW_BACKWARD, FFTW_ESTIMATE);

  M_i = malloc(N*N*N*sizeof(double));
  M_j = malloc(N*N*N*sizeof(double));
  g_i = malloc(N*N*N*sizeof(double));
  g_j = malloc(N*N*N*sizeof(double));
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


//Deallocator function
void dealloc_coll() {
  fftw_free(fftIn_f);
  fftw_free(fftOut_f);
  fftw_free(fftIn_g);
  fftw_free(fftOut_g);
  fftw_free(qHat);
  fftw_free(temp);
  free(wtN);
}

static void find_maxwellians(double *M_mat, double *g_mat, double *mat) {
  int i, j, k, index;
  double rho, vel[3], T, prefactor;

  rho = getDensity(mat, 0);
  getBulkVelocity(mat, vel, rho, 0);
  T = getTemperature(mat, vel, rho, 0);
  prefactor = rho * pow(0.5 / (M_PI * T), 1.5);
  for (index = 0; index < N * N * N; index++) {
    j = index / (N * N);
    i = (index - j * N * N) / N;
    k = index - N * (i + N * j);
    M_mat[index] = prefactor * exp(-(0.5/T) *((v[i]-vel[0])*(v[i]-vel[0]) + (v[j]-vel[1])*(v[j]-vel[1]) + (v[k]-vel[2])*(v[k]-vel[2])));
    g_mat[index] = mat[index] - M_i[index];
  }
}

static void compute_Qhat(double *f_mat, double *g_mat, int weightgenFlag, ...) {
  int index, x, y, z;
  double **conv_weights, *conv_weight_chunk;

  if (weightgenFlag == 0) {
     va_list args;
     va_start(args, weightgenFlag);
     conv_weights = va_arg(args, double **);
     va_end(args);
   }

  //#pragma omp parallel for private(qHat, fftIn_f, fftIn_g)
  for (index = 0; index < N * N * N; index++) {
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = f_mat[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = g_mat[index];
    fftIn_g[index][1] = 0.0;
  }

  //move to fourier space
  fft3D(fftIn_f, fftOut_f, noinverse);
  fft3D(fftIn_g, fftOut_g, noinverse);

  int zeta, zeta_x, zeta_y, zeta_z;
  int xi, xi_x, xi_y, xi_z;
  double cweight, prefactor;
  if (weightgenFlag == 1) {
    prefactor = 0.0625 * (diam_i + diam_j) * (diam_i + diam_j);
  }
  //#pragma omp parallel for private(zeta_x, zeta_y, zeta_z, xi_x, xi_y, xi_z, x, y, z, index, conv_weight_chunk, cweight)
  for (zeta = 0; zeta < N * N * N; zeta++) {
    zeta_x = zeta / (N * N);
    zeta_y = (zeta - zeta_x * N * N) / N;
    zeta_z = zeta - N * (zeta_y + zeta_x * N);
    if (weightgenFlag == 0) {
      conv_weight_chunk = conv_weights[zeta];
    }

    int n2 = N / 2;

  // #pragma omp simd
   for(xi = 0; xi < N * N * N; xi++) {
     xi_x = xi / (N * N);
     xi_y = (xi - xi_x * N * N) / N;
     xi_z = xi - N * (xi_y + xi_x * N);

      x = zeta_x + n2 - xi_x;
      y = zeta_y + n2 - xi_y;
      z = zeta_z + n2 - xi_z;

      if (x < 0)
        x = N + x;
      else if (x > N-1)
        x = x - N;

      if (y < 0)
        y = N + y;
      else if (y > N-1)
        y = y - N;

      if (z < 0)
        z = N + z;
      else if (z > N-1)
        z = z - N;

      index = z + N * (y + N * x);

      if (weightgenFlag == 0) {
        cweight = conv_weight_chunk[xi];
      }
      else {
        //Assume iso-case
        cweight = wtN[xi_x]*wtN[xi_y]*wtN[xi_z]*0.25*pow(0.5*(diam_i+diam_j),2) * gHat3(eta[xi_x], eta[xi_y], eta[xi_z], eta[zeta_x], eta[zeta_y], eta[zeta_z]);
      }
      //multiply the weighted fourier coeff product
      qHat[zeta][0] += cweight * (fftOut_g[xi][0]*fftOut_f[index][0] - fftOut_g[xi][1]*fftOut_f[index][1]);
      qHat[zeta][1] += cweight * (fftOut_g[xi][0]*fftOut_f[index][1] + fftOut_g[xi][1]*fftOut_f[index][0]);
    }
  }

  //End of parallel section
  fft3D(qHat, fftOut_f, inverse);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*
  function ComputeQ
  -----------------
  The main function for calculating the collision effects
*/
void ComputeQ_maxPreserve(double *f, double *g, double *Q, int weightgenFlag, ...) {
  double **conv_weights;
  if (weightgenFlag == 0) {
    va_list args;
    va_start(args, weightgenFlag);
    conv_weights = va_arg(args, double **);
    va_end(args);
  }

  int index;

  find_maxwellians(M_i, g_i, f);
  find_maxwellians(M_j, g_j, g);

  compute_Qhat(M_i, g_j, weightgenFlag, conv_weights);
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }

  compute_Qhat(g_i, M_j, weightgenFlag, conv_weights);
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
   Q[index] += fftOut_f[index][0];
  }

  compute_Qhat(g_i, g_j, weightgenFlag, conv_weights);
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
	Q[index] += fftOut_f[index][0];
  }

  //Maxwellian part
  /*
  compute_Qhat(M_i, M_j, conv_weights);
  //set Collision output
  for (index = 0; index < N * N * N; index++) {
    Q[index] += fftOut_f[index][0];
  }
  */
}

void ComputeQ(double *f, double *g, double *Q, int weightgenFlag, ...) {

  double **conv_weights;
  if (weightgenFlag == 0) {
    va_list args;
    va_start(args, weightgenFlag);
    conv_weights = va_arg(args, double **);
    va_end(args);
  }
  compute_Qhat(f, g, weightgenFlag, conv_weights);

  int index;
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


/*
function fft3D
--------------
Computes the fourier transform of in, and adjusts the coefficients based on our v, eta grids
*/
void fft3D(fftw_complex *in, fftw_complex *out, int invert) {
  int i, j, k, index;
  double sum, prefactor, factor;
  double delta, L_start, L_end, sign, *varr;
  fftw_plan p;

  if (invert == noinverse) {
    delta = dv;
    L_start = L_eta;
    L_end = L_v;
    varr = eta;
    sign = 1.0;
    p = p_forward;
  }
  else {
    delta = deta;
    L_start = L_v;
    L_end = L_eta;
    varr = v;
    sign = -1.0;
    p = p_backward;
  }
  prefactor = scale3 * delta * delta * delta;

  //shift the 'v' terms in the exponential to reflect our velocity domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * (double)(i + j + k) * L_start * delta;

    factor = prefactor * wtN[i] * wtN[j] * wtN[k];

    //dv correspond to the velocity space scaling - ensures that the FFT is properly scaled since fftw does no scaling at all
    temp[index][0] = factor * (cos(sum)*in[index][0] - sin(sum)*in[index][1]);
    temp[index][1] = factor * (cos(sum)*in[index][1] + sin(sum)*in[index][0]);
  }
  //computes fft
  fftw_execute(p);

  //shifts the 'eta' terms to reflect our fourier domain
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    sum = sign * L_end * (varr[i] + varr[j] + varr[k]);

    out[index][0] = cos(sum)*temp[index][0] - sin(sum)*temp[index][1];
    out[index][1] = cos(sum)*temp[index][1] + sin(sum)*temp[index][0];
  }

}

