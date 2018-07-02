#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <omp.h>

#include "constants.h"
#include "collisions.h"
#include "conserve.h"
#include "momentRoutines.h"

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

static void find_maxwellians(double *M_mat, double *g_mat, double *mat);
static void compute_Qhat(double *Q, double **conv_weights);

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
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + N * i);
    M_mat[index] = prefactor * exp(-(0.5/T) *((v[i]-vel[0])*(v[i]-vel[0]) + (v[j]-vel[1])*(v[j]-vel[1]) + (v[k]-vel[2])*(v[k]-vel[2])));
    g_mat[index] = mat[index] - M_i[index];
  }
}

static void compute_Qhat(double *Q, double **conv_weights) {
  int i, j, k, index, l, m, n, x, y, z;
  double *conv_weight_chunk;

  for (index = 0; index < N * N * N; index++) {
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = M_i[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = g_j[index];
    fftIn_g[index][1] = 0.0;
  }

  //move to foureir space
  fft3D(fftIn_f, fftOut_f);
  fft3D(fftIn_g, fftOut_g);

  #pragma omp parallel for private(i,j,k,l,m,n,x,y,z,conv_weight_chunk)
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    conv_weight_chunk = conv_weights[index];

    int n2 = N / 2;

    for(l=0;l<N;l++) {
      x = i + n2 - l;
      if (x < 0)
        x = N + x;
      else if (x > N-1)
        x = x - N;

      for(m=0;m<N;m++) {
        y = j + n2 - m;

        if (y < 0)
          y = N + y;
        else if (y > N-1)
          y = y - N;

        for(n=0;n<N;n++) {
          z = k + n2 - n;

          if (z < 0)
            z = N + z;
          else if (z > N-1)
            z = z - N;

      //multiply the weighted fourier coeff product
      qHat[k + N*(j + N*i)][0] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
          qHat[k + N*(j + N*i)][1] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);
    }
  }}}

  //End of parallel section
  ifft3D(qHat, fftOut_f);

  //set Collision output
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
void ComputeQ_maxPreserve(double *f, double *g, double *Q, double **conv_weights)
{
  int i, j, k, l, m, n, x, y, z, index, index1;
  double *conv_weight_chunk;

  double rho, vel[3], T, prefactor;

  find_maxwellians(M_i, g_i, f);
  find_maxwellians(M_j, g_j, g);

  compute_Qhat(Q, conv_weights);

  for (index = 0; index < N * N * N; index++) {
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = g_i[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = M_j[index];
    fftIn_g[index][1] = 0.0;
  }

  //move to fourier space
  fft3D(fftIn_f, fftOut_f);
  fft3D(fftIn_g, fftOut_g);


  #pragma omp parallel for private(i,j,k,l,m,n,x,y,z,conv_weight_chunk)
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    conv_weight_chunk = conv_weights[index];

    int n2 = N / 2;
    for(l=0;l<N;l++) {
      x = i + n2 - l;
      if (x < 0)
	x = N + x;
      else if (x > N-1)
	x = x - N;

      for(m=0;m<N;m++) {
	y = j + n2 - m;

	if (y < 0)
	  y = N + y;
	else if (y > N-1)
	  y = y - N;

	for(n=0;n<N;n++) {
	  z = k + n2 - n;

	  if (z < 0)
	    z = N + z;
	  else if (z > N-1)
	    z = z - N;

	  //multiply the weighted fourier coeff product
	  qHat[index][0] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
	  qHat[index][1] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);
	}
      }
    }
  }

  //End of parallel section

  ifft3D(qHat, fftOut_f);
  //set Collision output
  for (index = 0; index < N * N * N; index++) {
   Q[index] += fftOut_f[index][0];
  //second part - quadratic deviations
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = g_i[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = g_j[index];
    fftIn_g[index][1] = 0.0;
  }

  //move to fourier space
  fft3D(fftIn_f, fftOut_f);
  fft3D(fftIn_g, fftOut_g);


  #pragma omp parallel for private(i,j,k,l,m,n,x,y,z,conv_weight_chunk)
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    conv_weight_chunk = conv_weights[index];

    int n2 = N / 2;

    for(l=0;l<N;l++) {
      x = i + n2 - l;
      if (x < 0)
	x = N + x;
      else if (x > N-1)
	x = x - N;

      for(m=0;m<N;m++) {
	y = j + n2 - m;

	if (y < 0)
	  y = N + y;
	else if (y > N-1)
	  y = y - N;

	for(n=0;n<N;n++) {
	  z = k + n2 - n;

	  if (z < 0)
	    z = N + z;
	  else if (z > N-1)
	    z = z - N;

	  //multiply the weighted fourier coeff product
	  qHat[index][0] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
	  qHat[index][1] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);
	}
      }
    }
  }

  //End of parallel section

  ifft3D(qHat, fftOut_f);

  //set Collision output
  for (index = 0; index < N * N * N; index++) {
	Q[index] += fftOut_f[index][0];
  }

  //Maxwellian part
  /*
  for (index = 0; index < N * N * N; index++) {
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn_f[index][0] = M_i[index];
    fftIn_f[index][1] = 0.0;
    fftIn_g[index][0] = M_j[index];
    fftIn_g[index][1] = 0.0;
  }

  //move to fourier space
  fft3D(fftIn_f, fftOut_f);
  fft3D(fftIn_g, fftOut_g);


  #pragma omp parallel for private(i,j,k,l,m,n,x,y,z,conv_weight_chunk)
  for (index = 0; index < N * N * N; index++) {
    i = index / (N * N);
    j = (index - i * N * N) / N;
    k = index - N * (j + i * N);
    conv_weight_chunk = conv_weights[index];

    int n2 = N / 2;

    for(l=0;l<N;l++) {
      x = i + n2 - l;
      if (x < 0)
	x = N + x;
      else if (x > N-1)
	x = x - N;

      for(m=0;m<N;m++) {
	y = j + n2 - m;

	if (y < 0)
	  y = N + y;
	else if (y > N-1)
	  y = y - N;

	for(n=0;n<N;n++) {
	  z = k + n2 - n;

	  if (z < 0)
	    z = N + z;
	  else if (z > N-1)
	    z = z - N;

	  //multiply the weighted fourier coeff product
	  qHat[index][0] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
	  qHat[index][1] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);
	}
      }
    }
  }

  //End of parallel section

  ifft3D(qHat, fftOut_f);

  //set Collision output
  for (index = 0; index < N * N * N; index++) {
    Q[index] += fftOut_f[index][0];
  }
  */
}

void ComputeQ(double *f, double *g, double *Q, double **conv_weights)
{
  int i, j, k, l, m, n, x, y, z;
  int start_i, start_j, start_k, end_i, end_j, end_k;
  double *conv_weight_chunk;

  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)
	{
	     qHat[k + N*(j + N*i)][0] = 0.0;
	     qHat[k + N*(j + N*i)][1] = 0.0;
	  fftIn_f[k + N*(j + N*i)][0] = f[k + N*(j + N*i)];
	  fftIn_f[k + N*(j + N*i)][1] = 0.0;
	  fftIn_g[k + N*(j + N*i)][0] = g[k + N*(j + N*i)];
	  fftIn_g[k + N*(j + N*i)][1] = 0.0;
	}

  //move to fourier space
  fft3D(fftIn_f, fftOut_f);
  fft3D(fftIn_g, fftOut_g);


  #pragma omp parallel for private(i,j,k,l,m,n,x,y,z,start_i,start_j,start_k,end_i,end_j,end_k,conv_weight_chunk)
  for(i=0;i<N;i++)
  for(j=0;j<N;j++)
  for(k=0;k<N;k++) {

    conv_weight_chunk = conv_weights[k + N*(j + N*i)];

    int n2,n3;
    if(N % 2 == 0) {
      n2 = N/2;
      n3 = 1;
    }
    else {
      n2 = (N-1)/2;
      n3 = 0;
    }

    //figure out the windows for the convolutions (i.e. where xi(l) and eta(i)-xi(l) are in the domain)
    if( i < N/2 ) {
      start_i = 0;
      end_i = i + n2 + 1;
    }
    else {
      start_i = i - n2 + n3;
      end_i = N;
    }

    if( j < N/2 ) {
      start_j = 0;
      end_j = j + n2 + 1;
    }
    else {
      start_j = j - n2 + n3;
      end_j = N;
    }

    if( k < N/2 ) {
      start_k = 0;
      end_k = k + n2 + 1;
    }
    else {
      start_k = k - n2 + n3;
      end_k = N;
    }

    //no aliasing
    /*
    for(l=start_i;l<end_i;l++) {
      x = i + n2 - l;
      for(m=start_j;m<end_j;m++) {
	y = j + n2 - m;
	for(n=start_k;n<end_k;n++) {
	  z = k + n2 - n;
    */
    //aliasing

    for(l=0;l<N;l++) {
      x = i + n2 - l;
      if (x < 0)
	x = N + x;
      else if (x > N-1)
	x = x - N;

      for(m=0;m<N;m++) {
	y = j + n2 - m;

	if (y < 0)
	  y = N + y;
	else if (y > N-1)
	  y = y - N;

	for(n=0;n<N;n++) {
	  z = k + n2 - n;

	  if (z < 0)
	    z = N + z;
	  else if (z > N-1)
	    z = z - N;

	  //multiply the weighted fourier coeff product
	  qHat[k + N*(j + N*i)][0] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
	  qHat[k + N*(j + N*i)][1] += conv_weight_chunk[n + N*(m + N*l)]*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);

	  //Timing purposes only
	  //qHat[k + N*(j + N*i)][0] += 0.0*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][0] - fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][1]);
	  //qHat[k + N*(j + N*i)][1] += 0.0*(fftOut_g[n + N*(m + N*l)][0]*fftOut_f[z + N*(y + N*x)][1] + fftOut_g[n + N*(m + N*l)][1]*fftOut_f[z + N*(y + N*x)][0]);
	}
      }
    }
  }

  //End of parallel section

  ifft3D(qHat, fftOut_f);

  //set Collision output
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)	{
	Q[k + N*(j + N*i)] = fftOut_f[k + N*(j + N*i)][0];
      }
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


/*
function fft3D
--------------
Computes the fourier transform of in, and adjusts the coefficients based on our v, eta grids
*/
void fft3D(fftw_complex *in, fftw_complex *out)
{
  int i, j, k, index;
  double sum, prefactor;

  prefactor = scale3*dv*dv*dv;

  //shift the 'v' terms in the exponential to reflect our velocity domain
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)	{
	  index = k + N*(j + N*i);
	  sum = (double)(i + j + k) * L_eta*dv;

	  //dv correspond to the velocity space scaling - ensures that the FFT is properly scaled since fftw does no scaling at all
	  temp[index][0] = prefactor*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][0] - sin(sum)*in[index][1]);
	  temp[index][1] = prefactor*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][1] + sin(sum)*in[index][0]);
      }
  //computes fft
  fftw_execute(p_forward);

  //shifts the 'eta' terms to reflect our fourier domain
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	  index = k + N*(j + N*i);
	  sum = L_v*(eta[i] + eta[j] + eta[k]);

	  out[index][0] = cos(sum)*temp[index][0] - sin(sum)*temp[index][1];
	  out[index][1] = cos(sum)*temp[index][1] + sin(sum)*temp[index][0];
      }

}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

/*
function ifft3D
--------------
Computes the inverse fourier transform of in, and adjusts the coefficients based on our v, eta grid
*/

void ifft3D(fftw_complex *in, fftw_complex *out)
{
  int i, j, k, index;
  double sum;
  double prefactor = deta*deta*deta*scale3;

  //shifts the 'eta' terms to reflect our fourier domain
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	  index = k + N*(j + N*i);
	  sum = (double)(i + j + k)*L_v*deta*-1.0;

	  //deta ensures FFT is scaled correctly, since fftw does no scaling at all
	  temp[index][0] = prefactor*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][0] - sin(sum)*in[index][1]);
	  temp[index][1] = prefactor*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][1] + sin(sum)*in[index][0]);
      }
  //compute IFFT
  fftw_execute(p_backward);

  //shifts the 'v' terms to reflect our velocity domain
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) {
	  index = k + N*(j + N*i);
	  sum = -L_eta*(v[i] + v[j] + v[k]);

	  out[index][0] = cos(sum)*temp[index][0] - sin(sum)*temp[index][1];
	  out[index][1] = cos(sum)*temp[index][1] + sin(sum)*temp[index][0];
      }
}
