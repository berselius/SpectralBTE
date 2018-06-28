#include "conserve.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "species.h"

static double **CCt; //Need to go back to modify for different conservations if needed
static double *pivot;
static int N;
static double *wtN;
static double *v;
static double dv;
static int Ns;
static species *mixture;
static double *masterQ, *temp_cons;

void initialize_conservation(int nodes, double h_v, double *vel, species *mix, int numspec, int fast) {
    int i;

    N = nodes;
    dv = h_v;
    v = vel;

    if (fast == 0) {
        mixture = mix;
        Ns = numspec;
    }
    else {
        mixture = malloc(sizeof(species));
        Ns = 1;
        mixture[0].mass = 1.0;
    }

    CCt = malloc((Ns+4)*sizeof(double *));
    for(i=0;i<(Ns+4);i++) {
        CCt[i] = malloc((Ns+4)*sizeof(double));
    }
                 
    pivot = malloc((Ns+4)*sizeof(double));

    masterQ = malloc((Ns+4)*sizeof(double));
    temp_cons = malloc((Ns+4)*sizeof(double));

    wtN = malloc(N*sizeof(double));
    wtN[0] = 0.5;
    #pragma omp simd
    for(i=1;i<(N-1);i++) {
        wtN[i] = 1.0;
    }
    wtN[N-1] = 0.5;

    createCCtAndPivot();

}

void dealloc_conservation() {
    int i;
    for(i=0;i<Ns;i++) {
        free(CCt[i]);
    }
    free(CCt);
    free(masterQ);
    free(temp_cons);
    free(pivot);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
/* From Kendall Atkinson - pg 520 */

void factorMatrixIntoLU(int nElem, double det) {
    double *s, *c, *m, temp;
    int i, j, k, i0, ctr;
    
    s = malloc(nElem*sizeof(double));
    c = malloc(nElem*sizeof(double));
    m = malloc(nElem*sizeof(double));    
    
    det = 1.0;
    #pragma omp parallel for private (i, j, s)
    for(i=0;i<nElem;i++) {
        for(j=0;j<nElem;j++) {
            s[i] = fabs(CCt[i][0]);
            if(s[i] < fabs(CCt[i][j])) {
                s[i] = fabs(CCt[i][j]);
            }
        }
    }
    #pragma omp parallel for
    for(k=0;k<nElem-1;k++)
    {
        c[k] = fabs(CCt[k][k]/s[k]);
        ctr = 0;
        i0 = k;
        #pragma omp simd    
        for(i=k;i<nElem;i++)
        {
            if(c[k] < fabs(CCt[i][k]/s[i]))
            {
                c[k] = fabs(CCt[i][k]/s[i]);
                if(ctr != 1)
                {
                    i0 = i;
                    ctr = 1;
                }
            }
        }

        pivot[k] = i0;

        if(c[k] == 0.0)
        {
            det = 0.0;
            puts("Exiting factorMatrixintoLU....");
            exit(0);
        }
        
        if(i0 != k) {
            det = -det;
            #pragma omp simd
            for(j=k;j<nElem;j++)
            {
                temp = CCt[k][j];
                CCt[k][j] = CCt[i0][j];
                CCt[i0][j] = temp;
            }
            temp = s[k];
            s[k] = s[i0];
            s[i0] = temp;
        }

        #pragma omp simd
        for(i=k+1;i<nElem;i++) {
            for(j=k+1; j < nElem; j++) {
                m[i] = CCt[i][k]/CCt[k][k];
                CCt[i][k] = m[i];
                CCt[i][j] -= m[i]*CCt[k][j];
            }
        }

        det *= CCt[k][k];
    }
    
    det *= CCt[nElem-1][nElem-1];
    
    free(s);
    free(c);
    free(m);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
/* From Kendall Atkinson - pg 521 */

void solveWithCCt(int nElem, double *b)
{
    double temp, sum = 0.0;
    int i, j, k;
    
    #pragma omp parallel for
    for(k=0;k<nElem-1;k++)
    {
        i = pivot[k];
        if(pivot[k] != k)
        {
            temp = b[i];
            b[i] = b[k];
            b[k] = temp;
        }
        #pragma omp simd
        for(i=k+1;i<nElem;i++) {
            b[i] -= CCt[i][k]*b[k];
        }
    }
    
    b[nElem-1] = b[nElem-1]/CCt[nElem-1][nElem-1];
    #pragma omp parallel for
    for(i=nElem-2;i>=0;i--)
    {
        sum = 0.0;
        #pragma omp simd reduction (+:sum)
        for(j=i+1;j<nElem;j++) {
            sum += CCt[i][j]*b[j];
        }
        
        b[i] = 1.0/CCt[i][i]*(b[i] - sum);
    }
}    

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

//MAIN CONSERVATION ROUTINE
void conserveAllMoments(double **Q)
{
    int i, j, k, l, m, n, index;
    double prefactor, vi, vj, vk;
    double dv3 = dv * dv * dv;

    //lambda from before is now the sum_i sum_j C_i*Q_ij term
    #pragma omp simd
    for(l=0;l<(Ns+4);l++) {
        masterQ[l] = 0.0;
    }

    //computes sum_i sum_j C_i * Qij
    #pragma omp parallel for
    for (index = 0; index < N * N * N; index++) {
        i = index / (N * N);
        j = (index - i * N * N) / N;
        k = index - i * N * N - j * N;
        for(m=0;m<Ns;m++) {
            //entries of C_i
            prefactor = wtN[i] * wtN[j] * wtN[k] * dv3 * mixture[m].mass;
            vi = v[i];
            vj = v[j];
            vk = v[k];

            temp_cons[m] = prefactor;
            temp_cons[Ns] = prefactor * vi;
            temp_cons[Ns+1] = prefactor * vj;
            temp_cons[Ns+2] = prefactor * vk;
            temp_cons[Ns+3] = prefactor * 0.5 * (vi * vi + vj * vj + vk * vk);

            //computes sum_j    C_i * Qij
            #pragma omp simd
            for(n=0;n<Ns;n++) {
                masterQ[m] += Q[n*Ns + m][index] * temp_cons[m];
                masterQ[Ns] += Q[n*Ns + m][index] * temp_cons[Ns];
                masterQ[Ns+1] += Q[n*Ns + m][index] * temp_cons[Ns+1];
                masterQ[Ns+2] += Q[n*Ns + m][index] * temp_cons[Ns+2];
                masterQ[Ns+3] += Q[n*Ns + m][index] * temp_cons[Ns+3];            
            }
        }
    }
    

    solveWithCCt(Ns+4, masterQ);
    //masterQ now has the multipliers lambda = (sum_i C_i C_i^T)^(-1) * sum_i sum_j C_i Q_ij
    #pragma omp parallel for
    for (index = 0; index < N * N * N; index++) {
        i = index / (N * N);
        j = (index - i * N * N) / N;
        k = index - i * N * N - j * N;
        for(m=0;m<Ns;m++) {
            //entries of C_i^T
            prefactor = wtN[i] * wtN[j] * wtN[k] * dv3 * mixture[m].mass / Ns;
            vi = v[i];
            vj = v[j];
            vk = v[k];

            temp_cons[m] = prefactor;
            temp_cons[Ns] = prefactor * vi;
            temp_cons[Ns+1] = prefactor * vj;
            temp_cons[Ns+2] = prefactor * vk;
            temp_cons[Ns+3] = prefactor * 0.5 * (vi * vi + vj * vj + vk * vk);

            //Computes Q_ij = Q_ij - C_i^T lambda
            #pragma omp simd
            for(n=0;n<Ns;n++) {
                Q[n*Ns + m][index] -= (temp_cons[m]*masterQ[m] + temp_cons[Ns]*masterQ[Ns] + temp_cons[Ns+1]*masterQ[Ns+1] + temp_cons[Ns+2]*masterQ[Ns+2] + temp_cons[Ns+3]*masterQ[Ns+3]);
            }
        }
    }
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/

void createCCtAndPivot()
{
    double ***C, det = 1, prefactor;
    int i, j, k, m, n, index;
    double dv3 = dv * dv * dv;
    double vi, vj, vk;
    int N3 = N * N * N;
    
    //array of integration matrices
    C = malloc(Ns*sizeof(double **));
    for(i=0;i<Ns;i++) {
        C[i] = malloc((Ns+4)*sizeof(double *));
        for(j=0;j<(Ns+4);j++) {
            C[i][j] = malloc(N3 * sizeof(double));
        }
    }

    //Calculate each C_i
    #pragma omp parallel for
    for (index = 0; index < N3; index++) {
        i = index / (N * N);
        j = (index - i * N * N) / N;
        k = index - i * N * N - j * N;
        for(m=0;m<Ns;m++) {
            prefactor = wtN[i] * wtN[j] * wtN[k] * dv3 * mixture[m].mass;

            vi = v[i];
            vj = v[j];
            vk = v[k];

            #pragma omp simd
            for(n=0;n<Ns;n++) {
                C[m][n][index] = 0;
            }
                    
            C[m][m][index] = prefactor;            
            C[m][Ns][index] = prefactor * vi;
            C[m][Ns+1][index] = prefactor * vj;
            C[m][Ns+2][index] = prefactor * vk;
            C[m][Ns+3][index] = prefactor * 0.5 * (vi * vi + vj * vj + vk * vk);
        }
    }
    
    //now calculate sum_i C_i*C_i^T
    #pragma omp parallel for
    for(i=0;i<Ns+4;i++) {
        for(j=0;j<Ns+4;j++) {
            CCt[i][j] = 0.0;
            #pragma omp simd
            for(m=0;m<Ns;m++) {
                for(k=0;k<N3;k++) {
                    CCt[i][j] += C[m][i][k] * C[m][j][k];
                }
            }
        }
    }
    
    
    for(i=0;i<Ns;i++) {
        for(j=0;j<Ns+4;j++) {
            free(C[i][j]);
        }
        free(C[i]);
    }
    free(C);    

    
    factorMatrixIntoLU(Ns+4, det);
}
