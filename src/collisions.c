#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <omp.h>

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

static int noinverse = 0;
static int inverse = 1;

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
    
    wtN = malloc(N * sizeof(double));
    wtN[0] = 0.5;
    for(i = 1; i < (N-1); i++) {
        wtN[i] = 1.0;
    }
    wtN[N-1] = 0.5;

    //SETTING UP FFTW

    //allocate bins for ffts
    int N3 = N * N * N;
    fftIn_f = fftw_malloc(N3 * sizeof(fftw_complex));
    fftOut_f = fftw_malloc(N3 *sizeof(fftw_complex));
    fftIn_g = fftw_malloc(N3 * sizeof(fftw_complex));
    fftOut_g = fftw_malloc(N3 * sizeof(fftw_complex));
    qHat = fftw_malloc(N3 * sizeof(fftw_complex));
    temp = fftw_malloc(N3 * sizeof(fftw_complex));

    //Set up plans for FFTs
    p_forward    = fftw_plan_dft_3d (N, N, N, temp, temp, FFTW_FORWARD , FFTW_ESTIMATE);
    p_backward = fftw_plan_dft_3d (N, N, N, temp, temp, FFTW_BACKWARD, FFTW_ESTIMATE);

    M_i = malloc(N3 * sizeof(double));
    M_j = malloc(N3 * sizeof(double));
    g_i = malloc(N3 * sizeof(double));
    g_j = malloc(N3 * sizeof(double));
}


/*************************************************************************************************/


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


/*************************************************************************************************/

/*
    function ComputeQ
    -----------------
    The main function for calculating the collision effects
*/
void ComputeQ_maxPreserve(double *f, double *g, double *Q, double **conv_weights)
{ 
    double rho, vel[3], T;
    //Find Maxwellians

    find_maxwellians(&rho, vel, &T, f, M_i, g_i);
    find_maxwellians(&rho, vel, &T, g, M_j, g_j);

    ComputeQ(M_i, g_j, Q, conv_weights);
    ComputeQ(g_i, M_j, Q, conv_weights);
    //second part - quadratic deviations
    ComputeQ(g_i, g_j, Q, conv_weights);

    //Maxwellian part
    /*
    ComputeQ(M_i, M_j, Q, conv_weights);
    */        

}

static void find_maxwellians(double *rho, double vel[3], double *T, double *mat, double *M_mat, double *g_mat) {

    *rho = getDensity(mat, 0);
    getBulkVelocity(mat, vel, *rho, 0);
    *T = getTemperature(mat, vel, *rho, 0);

    double factor = pow(0.5 / M_PI * *T, 1.5) * *rho;
    double invT = -0.5 / *T;

    int index, i, j, k;
    double viv, vjv, vkv;
    #pragma omp parallel for collapse(2) private(i, j, k, index, viv, vjv, vkv)
    for(i=0;i<N;i++) {
        for(j=0;j<N;j++) {
            #pragma omp simd
            for(k=0;k<N;k++) {
                viv = v[i] - vel[0];
                vjv = v[j] - vel[1];
                vkv = v[k] - vel[2];
                index = k + N * (j + N * i);
                M_mat[index] = factor * exp(invT * (viv * viv + vjv * vjv + vkv * vkv));
                g_mat[index] = mat[index] - M_mat[index];
            }
        }
    }
}

//qopt report = 5
void ComputeQ(double *f, double *g, double *Q, double **conv_weights) {

    int index, index1, index2, i, j, k, l, m, n, x, y, z, n2;

    #pragma omp parallel for collapse(2) private (i, j, k, index)
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            #pragma omp simd
            for(k = 0; k < N; k++) {
                index = k + N * (j + N * i);
                qHat[index][0] = 0.0;
                qHat[index][1] = 0.0;
                fftIn_f[index][0] = f[index];
                fftIn_f[index][1] = 0.0;
                fftIn_g[index][0] = g[index];
                fftIn_g[index][1] = 0.0;
            }
        }
    }

    // Move to Fourier space
    fft3D(fftIn_f, fftOut_f, noinverse);
    fft3D(fftIn_g, fftOut_g, noinverse);
    
    double *conv_weight_chunk;

    n2 = N >> 1; // This is equivalent to n2 = N / 2

    double fftg0, fftg1, fftf0, fftf1, cweight;

    #pragma omp parallel for private(i, j, k, index, index1, index2, n2, l, m, n, x, y, z, fftg0, fftg1, fftf0, fftf1, cweight)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                index = k + N * (j + N * i);
                conv_weight_chunk = conv_weights[index];

                #pragma omp simd
                for (l = 0; l < N; l++) {
                    for (m = 0; m < N; m++) {
                        for (n = 0; n < N; n++) {
                            x = i + n2 - l;
                            y = j + n2 - l;
                            z = k + n2 - l;

                            if (x < 0)
                                x += N;
                            else if (x > N - 1)
                                x -= N;

                            if (y < 0)
                                y += N;
                            else if (y > N - 1)
                                y -= N;

                            if (z < 0)
                                z += N;
                            else if (z > N - 1)
                                z -= N;

                            index1 = n + N * (m + N * l);
                            index2 = z + N * (y + N * x);
                            fftg0 = fftOut_g[index1][0];
                            fftg1 = fftOut_g[index1][1];
                            fftf0 = fftOut_f[index2][0];
                            fftf1 = fftOut_f[index2][1];
                            cweight = conv_weight_chunk[index1];

                            //multiply the weighted fourier coeff product        
                            qHat[index][0] += cweight * (fftg0 * fftf0 - fftg1 * fftf1);    
                            qHat[index][1] += cweight * (fftg0 * fftf1 + fftg1 * fftf0);
                        }
                    }
                }
            }
        }
    }
    // Exit Fourier space
    fft3D(qHat, fftOut_f, inverse);

    //set Collision output
    #pragma omp simd private(index)
    for(i = 0; i < N; i++)
        for(j = 0 ;j < N; j++)
            for(k = 0; k < N; k++) {
                index = k + N * (j + N * i);
                Q[index] = fftOut_f[index][0];
            }
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


/*
Computes the (inverse) fourier transform of in, and adjusts the coefficients
based on v or eta grid.
*/
void fft3D(fftw_complex *in, fftw_complex *out, int inverse) {
    int i, j, k, index;
    double sum, prefactor, factor, mult0, mult1;
    double delta, offset, L, cosine, sine, sign, *var;
    fftw_plan p;

    if (inverse == 1) {
        delta = dv;
        offset = -1.0;
        L = L_eta;
        p = p_backward;
        sign = -1.0;
        var = eta;
    }
    else {
        delta = deta;
        offset = 0.0;
        L = L_v;
        p = p_forward;
        sign = 1.0;
        var = v;
    }

    prefactor = scale3 * delta * delta * delta;

    // Shift v (or eta) terms to reflect velocity (or fourier) domain
    #pragma omp parallel for private(i, j, k, index, sum, cosine, sine, factor, mult0, mult1)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            #pragma omp simd
            for (k = 0; k < N; k++) {
                index = k + N * (j + N * i);
                sum = (double)(i + j + k) * L * delta + offset;

                cosine = cos(sum);
                sine = sin(sum);
                factor = prefactor * wtN[i] * wtN[j] * wtN[k];
                mult0 = in[index][0];
                mult1 = in[index][1];

                temp[index][0] = factor * (cosine * mult0 - sine * mult1);
                temp[index][1] = factor * (cosine * mult1 + sine * mult0);
            }
    // Compute FFT or Inverse FFT
    fftw_execute(p);
    // shift eta (or v) terms to reflect fourier (or velocity) domain
    #pragma omp parallel for private(i, j, k, index, sum, cosine, sine, mult0, mult1)
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            for (k = 0; k < N; k++) {
                index = k * N * (j + N * i);
                sum = sign * L * (var[i] + var[j] + var[k]);

                sine = sin(sum);
                cosine = cos(sum);
                mult0 = temp[index][0];
                mult1 = temp[index][1];

                out[index][0] = cosine * mult0 - sine * mult1;
                out[index][1] = cosine * mult1 + sine * mult0;
            } 
}









