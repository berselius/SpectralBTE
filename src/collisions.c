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
#include "collisions_fft3d_cpu.h"
#include "collisions_fft3d_gpu.h"

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

static int cudaFlag = 0;

time_t function_time;

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

  find_maxwellians(M_i, g_i, f, M_i, v, N);
  find_maxwellians(M_j, g_j, g, M_i, v, N);

  compute_Qhat(M_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, dv, L_v, eta, deta, L_eta, wtN_global, N, scale3, cudaFlag, weightgenFlag, conv_weights);
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }

  compute_Qhat(g_i, M_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, dv, L_v, eta, deta, L_eta, wtN_global, N, scale3, cudaFlag, weightgenFlag, conv_weights);
  //set Collision output
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
   Q[index] += fftOut_f[index][0];
  }

  compute_Qhat(g_i, g_j, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, dv, L_v, eta, deta, L_eta, wtN_global, N, scale3, cudaFlag, weightgenFlag, conv_weights);
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

  compute_Qhat(f, g, qHat, fftIn_f, fftOut_f, fftIn_g, fftOut_g, v, dv, L_v, eta, deta, L_eta, wtN_global, N, scale3, cudaFlag, weightgenFlag, conv_weights);

  int index;
  //set Collision output
  function_time = clock();
  //#pragma omp parallel for private(Q)
  for (index = 0; index < N * N * N; index++) {
    Q[index] = fftOut_f[index][0];
  }
}
