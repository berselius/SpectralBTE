#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <stdarg.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "constants.h"
#include "collisions.h"
#include "conserve.h"
#include "momentRoutines.h"
#include "weights.h"

#include "collisions_support_cpu.h"
#include "collisions_support_gpu.h"

static double (*fftIn_f)[2], (*fftOut_f)[2], (*fftIn_g)[2], (*fftOut_g)[2], (*qHat)[2];
static double (*fftIn_f_cuda)[2], (*fftOut_f_cuda)[2], (*fftIn_g_cuda)[2], (*fftOut_g_cuda)[2], (*qHat_cuda)[2];
static double *M_i, *M_j, *g_i, *g_j;
static double L_v;
static double L_eta;
static double *v;
static double *eta;

static double *eta_cuda, *v_cuda;

static double dv;
static double deta;
static int N;
static double *wtN_global;

static double scale3;

static int cudaFlag = 1;

time_t function_time;

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

  scale3 = pow(1.0 / sqrt(2.0*M_PI), 3.0);
  initialize_collisions_support_cpu(N, scale3);

  wtN_global = malloc(N*sizeof(double));
  wtN_global[0] = 0.5;
  #pragma omp simd
  for(i=1;i<(N-1);i++) {
    wtN_global[i] = 1.0;
  }
  wtN_global[N-1] = 0.5;

  //SETTING UP FFTW

  //allocate bins for ffts
  fftIn_f = malloc(N*N*N*sizeof(double[2]));
  fftOut_f = malloc(N*N*N*sizeof(double[2]));
  fftIn_g = malloc(N*N*N*sizeof(double[2]));
  fftOut_g = malloc(N*N*N*sizeof(double[2]));
  qHat = malloc(N*N*N*sizeof(double[2]));


  if (cudaFlag == 1) {
    initialize_collisions_support_gpu(wtN_global, N);

    cudaMalloc((void**)&eta_cuda, N*sizeof(double));
    cudaMalloc((void**)&v_cuda, N*sizeof(double));
    cudaMemcpy(eta_cuda, eta, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_cuda, v, N*sizeof(double), cudaMemcpyHostToDevice);
  }

  M_i = malloc(N*N*N*sizeof(double));
  M_j = malloc(N*N*N*sizeof(double));
  g_i = malloc(N*N*N*sizeof(double));
  g_j = malloc(N*N*N*sizeof(double));
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


//Deallocator function
void dealloc_coll() {
  free(fftIn_f);
  free(fftOut_f);
  free(fftIn_g);
  free(fftOut_g);
  free(qHat);
  free(wtN_global);

  if (cudaFlag == 1) {
    cudaFree(eta_cuda);
    cudaFree(v_cuda);

    deallocate_collisions_support_gpu();
  }

  deallocate_collisions_support_cpu();
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


  // move to Fourier space
  if (cudaFlag == 0) { // Use CPU version
    fft3D_cpu(fftIn_f, fftOut_f, dv, L_eta, L_v, 1.0, eta, wtN_global);
    fft3D_cpu(fftIn_g, fftOut_g, dv, L_eta, L_v, 1.0, eta, wtN_global);
  }
  else { // Use GPU version
    fft3D_gpu(fftIn_f, fftOut_f, dv, L_eta, L_v, 1.0, eta_cuda, scale3, N);
    fft3D_gpu(fftIn_g, fftOut_g, dv, L_eta, L_v, 1.0, eta_cuda, scale3, N);
  }

  int zeta, zeta_x, zeta_y, zeta_z;
  int xi, xi_x, xi_y, xi_z;
  double cweight, prefactor;
  if (weightgenFlag == 1) {
    prefactor = 0.0625 * (diam_i + diam_j) * (diam_i + diam_j);
  }
  #pragma omp parallel for private(zeta_x, zeta_y, zeta_z, xi_x, xi_y, xi_z, x, y, z, index, conv_weight_chunk, cweight)
  for (zeta = 0; zeta < N * N * N; zeta++) {
    zeta_x = zeta / (N * N);
    zeta_y = (zeta - zeta_x * N * N) / N;
    zeta_z = zeta - N * (zeta_y + zeta_x * N);
    if (weightgenFlag == 0) {
      conv_weight_chunk = conv_weights[zeta];
    }

    int n2 = N / 2;

   #pragma omp simd
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
        cweight = wtN_global[xi_x] * wtN_global[xi_y] * wtN_global[xi_z] * prefactor * gHat3(eta[xi_x], eta[xi_y], eta[xi_z], eta[zeta_x], eta[zeta_y], eta[zeta_z]);
      }
      //multiply the weighted fourier coeff product
      qHat[zeta][0] += cweight * (fftOut_g[xi][0]*fftOut_f[index][0] - fftOut_g[xi][1]*fftOut_f[index][1]);
      qHat[zeta][1] += cweight * (fftOut_g[xi][0]*fftOut_f[index][1] + fftOut_g[xi][1]*fftOut_f[index][0]);
    }
  }

  //End of parallel section
  fft3D_cpu(qHat, fftOut_f, deta, L_v, L_eta, -1.0, v, wtN_global);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*
  function ComputeQ
  -----------------
  The main function for calculating the collision effects
*/
void ComputeQ_maxPreserve(double *f, double *g, double *Q, int weightgenFlag, ...) {
// timers = [Q_hat, Q_max_preserve, Q, maxwellian]
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
    compute_Qhat(f, g, weightgenFlag, conv_weights);
  }
  else {
    compute_Qhat(f, g, weightgenFlag);
  }

  int index;
  //set Collision output
  function_time = clock();
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
}
