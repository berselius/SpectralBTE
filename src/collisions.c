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

#include "collisions_cpu.h"
#include "collisions_gpu.h"

static double (*fftIn_f)[2], (*fftOut_f)[2], (*fftIn_g)[2], (*fftOut_g)[2], (*qHat)[2];
static double (*fftIn_f_cuda)[2], (*fftOut_f_cuda)[2], (*fftIn_g_cuda)[2], (*fftOut_g_cuda)[2], (*qHat_cuda)[2];
static double *M_i, *M_j, *g_i, *g_j;

struct FFTVars v;
struct FFTVars eta;
struct FFTVars v_cuda;
struct FFTVars eta_cuda;

static int N;
static double *wtN_global;

static double scale3;

static int cudaFlag = 1;

time_t function_time;

//Initializes this module's static variables and allocates what needs allocating
void initialize_coll(int nodes, double length, double *vel, double *zeta) {
  int i;

  N = nodes;
  v.L_var = length;
  v.var = vel;
  v.d_var = vel[1] - vel[0];

  eta.var = zeta;
  eta.d_var = zeta[1]-zeta[0];;
  eta.L_var = -zeta[0];
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

    v_cuda.L_var = v.L_var;
    v_cuda.d_var = v.d_var;
    eta_cuda.L_var = eta.L_var;
    eta_cuda.d_var = eta.d_var;

    cudaMalloc((void**)&eta_cuda.var, N*sizeof(double));
    cudaMalloc((void**)&v_cuda.var, N*sizeof(double));
    cudaMemcpy(eta_cuda.var, eta.var, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(v_cuda.var, v.var, N*sizeof(double), cudaMemcpyHostToDevice);
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
    cudaFree(eta_cuda.var);
    cudaFree(v_cuda.var);

    deallocate_collisions_support_gpu();
  }

  deallocate_collisions_support_cpu();
}

void ComputeQ_maxPreserve_gpu(double *f, double *g, double *Q) {
  find_maxwellians_gpu(M_i, g_i, f, M_i, v.var, N);
  find_maxwellians_gpu(M_j, g_j, g, M_i, v.var, N);
  int index;
  compute_Qhat_gpu(M_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3);
  //set collision output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
  compute_Qhat_gpu(g_i, M_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3);
  //set collisions output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
   Q[index] += fftOut_f[index][0];
  }
  compute_Qhat_gpu(g_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3);
  //set collisions output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
        Q[index] += fftOut_f[index][0];
  }
}

void ComputeQ_gpu(double *f, double *g, double *Q) {
  compute_Qhat_gpu(f, g, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3);

  int index;
  //set Collision output
  function_time = clock();
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*
  function ComputeQ
  -----------------
  The main function for calculating the collision effects
*/
void ComputeQ_maxPreserve_cpu(double *f, double *g, double *Q, int weightgenFlag, ...) {
// timers = [Q_hat, Q_max_preserve, Q, maxwellian]
  double **conv_weights;
  if (weightgenFlag == 0) {
    va_list args;
    va_start(args, weightgenFlag);
    conv_weights = va_arg(args, double **);
    va_end(args);
  }

  int index;

  find_maxwellians_cpu(M_i, g_i, f, M_i, v.var, N);
  find_maxwellians_cpu(M_j, g_j, g, M_i, v.var, N);

  compute_Qhat_cpu(M_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3, weightgenFlag, conv_weights);
  //set Collision output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
  compute_Qhat_cpu(g_i, M_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3, weightgenFlag, conv_weights);

  //set Collision output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
   Q[index] += fftOut_f[index][0];
  }
  compute_Qhat_cpu(g_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3, weightgenFlag, conv_weights);
  //set Collision output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
	Q[index] += fftOut_f[index][0];
  }
}

void ComputeQ_cpu(double *f, double *g, double *Q, int weightgenFlag, ...) {
  double **conv_weights;
  if (weightgenFlag == 0) {
    va_list args;
    va_start(args, weightgenFlag);
    conv_weights = va_arg(args, double **);
    va_end(args);
  }

  compute_Qhat_cpu(f, g, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, eta, wtN_global, N, scale3, weightgenFlag, conv_weights);

  int index;
  //set Collision output
  #pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
}
