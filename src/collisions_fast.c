#include <math.h>
#include <fftw3.h>
#include <stdlib.h>
#include <omp.h>
#include <complex.h>

#include "collisions_fast.h"
#include "conserve.h"
#include "momentRoutines.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include "gauss_legendre.h"

static double L_v;
static double L_eta;
static double *v;
static double *eta;
static double dv;
static double deta;
static int N;
static double *wtN;
static double scale3;
static double lambda;
static fftw_complex *fftIn, *fftOut, *qHat;

static fftw_plan p_forward; 
static fftw_plan p_backward; 
static fftw_plan p_forward_F; 
static fftw_plan p_forward_f; 
static fftw_plan p_backward_G; 
static fftw_complex *F, *Finv, *finv;
static fftw_complex *G, *Ginv;
static fftw_complex *temp;
static int aniso;

static double *r, *theta, *phi, *wtR, *wtTh, *wtPh;
static int M_r, M_th, M_ph;
static double **alphaWtRe, **alphaWtIm, **betaWtRe, **betaWtIm;

static  gsl_integration_cquad_workspace *w_beta;


/*$$$$$$$$$$$$$$$$$$$$$$$
Functions for computing the convolution quadrature weights
  $$$$$$$$$$$$$$$$$$$$$$$*/

double sinc(double x) {
  double ret;
  if (x != 0) 
    ret = sin(x) / x;
  else
    ret = 1.0;
  return ret;
	     
}

double betaIntegrandRe(double alpha, void *args) {
  double *dargs = (double *)args;

  double zetadot = sin(dargs[1])*(cos(dargs[2])*dargs[3] + sin(dargs[2])*dargs[4]) + cos(dargs[1])*dargs[5];

  double zetaperp_len = sqrt(dargs[3]*dargs[3] + dargs[4]*dargs[4] + dargs[5]*dargs[5] - zetadot*zetadot);

  return (sin(alpha) / pow(sin(0.5*alpha),4)) * ( cos(0.5*dargs[0]*(1-cos(alpha))*zetadot) * j0(0.5*dargs[0]*sin(alpha)*zetaperp_len) - 1);
}


double betaIntegrandIm(double alpha, void *args) {
  double *dargs = (double *)args;

  double zetadot = sin(dargs[1])*(cos(dargs[2])*dargs[3] + sin(dargs[2])*dargs[4]) + cos(dargs[1])*dargs[5];

  double zetaperp_len = sqrt(dargs[3]*dargs[3] + dargs[4]*dargs[4] + dargs[5]*dargs[5] - zetadot*zetadot);

  return (sin(alpha) / pow(sin(0.5*alpha),4)) * ( sin(0.5*dargs[0]*(1-cos(alpha))*zetadot) * j0(0.5*dargs[0]*sin(alpha)*zetaperp_len)    );
}

void alphaWt(double r, double theta, double phi, double *value_re, double *value_im) {
  int i,j,k;
  double zetadot;

  for(i=0;i<N;i++) 
    for(j=0;j<N;j++) 
      for(k=0;k<N;k++) {
	zetadot = sin(theta)*(eta[i]*cos(phi) + eta[j]*sin(phi)) + eta[k]*cos(theta);
	value_re[k + N*(j + N*i)] =      cos(r*zetadot);
	value_im[k + N*(j + N*i)] = -1.0*sin(r*zetadot);
      }
}

void betaWt(double r, double theta, double phi, double *value_re, double *value_im) {
  int i,j,k;
  double zetadot, zetalen, sincinput, sincval, zetaperp;
  double result_re, result_im;
  double c1, c2;

  gsl_function beta_Re;
  gsl_function beta_Im;
  double args[6];

  beta_Re.function = &betaIntegrandRe;
  beta_Re.params = args;
  beta_Im.function = &betaIntegrandIm;
  beta_Im.params = args;

  args[0] = r;
  args[1] = theta;
  args[2] = phi;

  if(!aniso) {

    for(i=0;i<N;i++)
      for(j=0;j<N;j++)
	for(k=0;k<N;k++) {
	  zetadot = sin(theta)*(eta[i]*cos(phi) + eta[j]*sin(phi)) + eta[k]*cos(theta);
	  zetalen = sqrt(eta[i]*eta[i] + eta[j]*eta[j] + eta[k]*eta[k]);
	  sincinput = 0.5*r*zetalen;
	  
	  sincval = sinc(sincinput);
 
	  value_re[k + N*(j + N*i)] = (cos(0.5*r*zetadot)*sincval - 1);
	  value_im[k + N*(j + N*i)] =  sin(0.5*r*zetadot)*sincval    ;
	}
  }
  else {
    //Aniso case
    for(i=0;i<N;i++)
      for(j=0;j<N;j++)
	for(k=0;k<N;k++) {
	  args[3] = eta[i];
	  args[4] = eta[j];
	  args[5] = eta[k];
	  gsl_integration_cquad(&beta_Re,0.01,M_PI,1e-6,1e-6,w_beta,&result_re,NULL,NULL);  //"good" part
	  gsl_integration_cquad(&beta_Im,0.01,M_PI,1e-6,1e-6,w_beta,&result_im,NULL,NULL);  //"good" part
	  value_re[k + N*(j + N*i)] = 2.0*M_PI*result_re/(-M_PI*log(sin(0.0001/2)));
	  value_im[k + N*(j + N*i)] = 2.0*M_PI*result_im/(-M_PI*log(sin(0.0001/2)));

	  zetadot = sin(theta)*(eta[i]*cos(phi) + eta[j]*sin(phi)) + eta[k]*cos(theta);

	  //The singular-ish part
	  c1 = 0.5*r*zetadot;

	  zetaperp = sqrt(args[3]*args[3] + args[4]*args[4] + args[5]*args[5] - zetadot*zetadot);

	  c2 = 0.5*r*zetaperp;

	  value_re[k + N*(j + N*i)] += 2.0*M_PI*-0.25*c2*c2*8.0*log(0.0001)/(M_PI*log(sin(0.0001/2)));
	  value_im[k + N*(j + N*i)] += 2.0*M_PI* 0.5*c1    *8.0*log(0.0001)/(M_PI*log(sin(0.0001/2)));	  
      }    
  }
}


//Initializes this module's static variables and allocates what needs allocating
void initialize_coll(int nodes, double length, double *vel, double *M, double *zeta, double lam) {
  int i,l,m,n;

  N = nodes;
  L_v = length;
  v = vel;
  dv = v[1]-v[0];
  lambda = lam;

  scale3 = pow(1.0/sqrt(2.0*M_PI), 3.0);
  //oneNcubed = 1.0/(N*N*N);
  
  wtN = malloc(N*sizeof(double));
  wtN[0] = 0.5;
  for(i=1;i<(N-1);i++)
    wtN[i] = 1.0;
  wtN[N-1] = 0.5;

  eta = zeta;
  deta = eta[1]-eta[0];
  L_eta = -eta[0];
  //L_eta = 0.0;

  aniso = 0;
  w_beta = gsl_integration_cquad_workspace_alloc(1000);

  //Allocate quadrature information (gauss-legendre)
  M_r = M[0];
  M_th = M[1];
  M_ph = M[2];

  /*  double dr = L_v / ((double) M_r-1.);
      double dth = M_PI /(double) M_th;
      double dphi = 2.0*M_PI / (double)M_ph;*/

  r = malloc(M_r*sizeof(double));
  wtR = malloc(M_r*sizeof(double));
  theta = malloc(M_th*sizeof(double));
  wtTh = malloc(M_th*sizeof(double));
  phi = malloc(M_ph*sizeof(double));
  wtPh = malloc(M_ph*sizeof(double));

  /*
  r[0]   = 0.0;
  wtR[0] = 0.5*dr;
  for(i=1;i<(M_r-1);i++){
    wtR[i] = 1.0*dr;
    r[i] = i*dr;
  }
  r[M_r-1]   = L_v;
  wtR[M_r-1] = 0.5*dr;
  */
  
  double *x = malloc(M_r*sizeof(double));
  double *wtx = malloc(M_r*sizeof(double));

  gauss_legendre_tbl(M_r,x,wtx,1e-10);
  double A = L_v/2.0; //width from midpoint
  double B = L_v/2.0; //midpoint

  m = M_r>>1;
  if(M_r&1) {//odd
    r[m] = B;
    wtR[m] = A*wtx[0];
    for(i=1;i<=m;i++) {
      r[m + i]       = B + A*x[i];
      r[m - i]       = B - A*x[i];
      wtR[m + i]     = A*wtx[i];
      wtR[m - i]     = A*wtx[i];
    }
  }
  else //even
    for(i=0;i<m;i++) {
      r[m+i]         = B + A*x[i];
      r[m-i-1]       = B - A*x[i];
      wtR[m+i]       = A*wtx[i];
      wtR[m-i-1]     = A*wtx[i];
    }
  
  free(x);
  free(wtx);

  /*
  if((M_r % 2) == 0)
    start = M_r/2;
  else
    start = M_r/2 + 1;
  //post process
  for(i=start;i<M_r;i++) {
    r[i]   =  -r[i-M_r/2];
    wtR[i] = wtR[i-M_r/2];
  }
  for(i=0;i<M_r;i++) {
    r[i]   = 0.5*L_v + 0.5*L_v*r[i];
    wtR[i] = 0.5*L_v*wtR[i];
  }
  */

  /*
  for(i=0;i<M_th;i++) {
    wtTh[i] = 1.0*dth;
    theta[i] = i*dth;
  }
  */

  x = malloc(M_th*sizeof(double));
  wtx = malloc(M_th*sizeof(double));

  gauss_legendre_tbl(M_th,x,wtx,1e-10);

  A = M_PI/2.0;
  B = M_PI/2.0;

  m = M_th>>1;
  if(M_th&1) {//odd
    theta[m] = B;
    wtTh[m] = A*wtx[0];
    for(i=1;i<=m;i++) {
      theta[m + i]       = B + A*x[i];
      theta[m - i]       = B - A*x[i];
      wtTh[m + i]     = A*wtx[i];
      wtTh[m - i]     = A*wtx[i];
    }
  }
  else //even
    for(i=0;i<m;i++) {
      theta[m+i]         = B + A*x[i];
      theta[m-i-1]       = B - A*x[i];
      wtTh[m+i]       = A*wtx[i];
      wtTh[m-i-1]     = A*wtx[i];
    }
  
  free(x);
  free(wtx);
  /*
  if((M_th % 2) == 0)
    start = M_th/2;
  else
    start = M_th/2 + 1;

  //post process
  for(i=start;i<M_th;i++) {
    theta[i] = -theta[i-M_th/2];
    wtTh[i]  =   wtTh[i-M_th/2];
  }
  for(i=0;i<M_th;i++) {
    theta[i] = 0.5*M_PI + 0.5*M_PI*theta[i];
    wtTh[i]  = 0.5*M_PI*wtTh[i];
  }
  */


  x = malloc(M_th*sizeof(double));
  wtx = malloc(M_th*sizeof(double));

  gauss_legendre_tbl(M_ph,x,wtx,1e-10);

  A = M_PI;
  B = M_PI;

  m = M_ph>>1;
  if(M_ph&1) {//odd
    phi[m] = B;
    wtPh[m] = A*wtx[0];
    for(i=1;i<=m;i++) {
      phi[m + i]       = B + A*x[i];
      phi[m - i]       = B - A*x[i];
      wtPh[m + i]     = A*wtx[i];
      wtPh[m - i]     = A*wtx[i];
    }
  }
  else //even
    for(i=0;i<m;i++) {
      phi[m+i]         = B + A*x[i];
      phi[m-i-1]       = B - A*x[i];
      wtPh[m+i]       = A*wtx[i];
      wtPh[m-i-1]     = A*wtx[i];
    }
  
  free(x);
  free(wtx);

  /*
  if((M_ph % 2) == 0)
    start = M_ph/2;
  else
    start = M_ph/2 + 1;

  //post process
  for(i=start;i<M_ph;i++) {
    phi[i]  = -phi[i-M_ph/2];
    wtPh[i] = wtPh[i-M_ph/2];
  }
  for(i=0;i<M_ph;i++) {
    phi[i]  = M_PI + M_PI*phi[i];
    wtPh[i] = M_PI*wtPh[i];
  }
  */

  /*
  dphi = 2.0*M_PI/M_ph;
  for(i=0;i<M_ph;i++) {
    phi[i]  = i*dphi;
    wtPh[i] = dphi;
  }
  */


  //SETTING UP FFTW

  fftw_init_threads();
  
  fftIn  = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  fftOut = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  qHat   = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  temp   = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  F      = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  G      = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  Finv   = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  finv   = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  Ginv   = (fftw_complex *)fftw_malloc(N*N*N*sizeof(fftw_complex));
  

  //Set up plans for FFTs
  
  fftw_plan_with_nthreads(omp_get_num_threads());

  p_forward    = fftw_plan_dft_3d (N, N, N, temp  , temp, FFTW_FORWARD , FFTW_MEASURE);
  p_backward   = fftw_plan_dft_3d (N, N, N, temp  , temp, FFTW_BACKWARD, FFTW_MEASURE);
  p_backward_G = fftw_plan_dft_3d (N, N, N, Ginv  , G   , FFTW_BACKWARD, FFTW_MEASURE);
  p_forward_F  = fftw_plan_dft_3d (N, N, N, F     , Finv, FFTW_FORWARD , FFTW_MEASURE); 
  p_forward_f  = fftw_plan_dft_3d (N, N, N, fftOut, finv, FFTW_FORWARD , FFTW_MEASURE); 
  
  FILE *fidWeights;
  char buffer_weights[100];
  size_t readFlag;

  alphaWtRe = malloc(M_r*M_th*M_ph*sizeof(double *));
  alphaWtIm = malloc(M_r*M_th*M_ph*sizeof(double *));
  betaWtRe  = malloc(M_r*M_th*M_ph*sizeof(double *));
  betaWtIm  = malloc(M_r*M_th*M_ph*sizeof(double *));
  
  
  //set up and fill weights
  for(l=0;l<M_r;l++)
    for(m=0;m<M_th;m++)
      for(n=0;n<M_ph;n++) {
  	alphaWtRe[n + M_ph*(m + M_th*l)] = malloc(N*N*N*sizeof(double));
	alphaWtIm[n + M_ph*(m + M_th*l)] = malloc(N*N*N*sizeof(double));
	 betaWtRe[n + M_ph*(m + M_th*l)] = malloc(N*N*N*sizeof(double));
	 betaWtIm[n + M_ph*(m + M_th*l)] = malloc(N*N*N*sizeof(double));
      }  
  
  sprintf(buffer_weights,"FastWeights/N%d_M%d_L_v%g_lambda%g.wts",N, (int)M_th, L_v,lambda);

  if((fidWeights = fopen(buffer_weights,"r"))) {
    printf("Loading weights from file %s\n",buffer_weights);
    for(l=0;l<M_r;l++)
      for(m=0;m<M_th;m++)
	for(n=0;n<M_ph;n++) {
	  readFlag = fread(alphaWtRe[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  if(readFlag != N*N*N) {
	    printf("Error reading weight file\n");
	    exit(1);
	  }
	  
	  
	  readFlag = fread(alphaWtIm[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  if(readFlag != N*N*N) {
	    printf("Error reading weight file\n");
	    exit(1);
	  }
	  

	  readFlag = fread( betaWtRe[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  if(readFlag != N*N*N) {
	    printf("Error reading weight file\n");
	    exit(1);
	  }
	  

	  readFlag = fread( betaWtIm[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  if(readFlag != N*N*N) {
	    printf("Error reading weight file\n");
	    exit(1);
	  }
	}
  }
  else {

    printf("Generating weights\n");
    
    
    //set up and fill weights
    for(l=0;l<M_r;l++)
      for(m=0;m<M_th;m++)
	for(n=0;n<M_ph;n++) {
	  alphaWt(r[l],theta[m],phi[n],alphaWtRe[n + M_ph*(m + M_th*l)],alphaWtIm[n + M_ph*(m + M_th*l)]);
	   betaWt(r[l],theta[m],phi[n], betaWtRe[n + M_ph*(m + M_th*l)], betaWtIm[n + M_ph*(m + M_th*l)]);
	}
    
    //store weights
    fidWeights = fopen(buffer_weights,"w");
    for(l=0;l<M_r;l++)
      for(m=0;m<M_th;m++)
	for(n=0;n<M_ph;n++) {
	  fwrite(alphaWtRe[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  fwrite(alphaWtIm[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  fwrite( betaWtRe[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	  fwrite( betaWtIm[n + M_ph*(m + M_th*l)],sizeof(double),N*N*N,fidWeights);
	} 
    if(fflush(fidWeights) != 0) {
      printf("Something is wrong with storing the weights");
      exit(0);
    }      
    
    printf("Done generating weights!\n");
  }
  
  fclose(fidWeights);
  
  //Set up conservation routines
  initialize_conservation_fast(N, dv, v);
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


//Deallocator function
void dealloc_coll() {
  fftw_free(fftIn);
  fftw_free(fftOut);
  fftw_free(qHat);
  fftw_free(temp);
  fftw_free(F);
  fftw_free(G);
  fftw_free(Finv);
  fftw_free(Ginv);
  free(eta);
}


/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/


void collision(double *f, double *Q) {
  ComputeQ(f,Q);
  conserveAllMoments(&Q);
}


/*
  function ComputeQ
  -----------------
  The main function for calculating the collision effects
*/
void ComputeQ(double *f, double *Q)
{
  int l, m, n;
  int index, index_quad;
  double prefac2;

  for(index=0;index<N*N*N;index++)	{
    qHat[index][0] = 0.0;
    qHat[index][1] = 0.0;
    fftIn[index][0] = f[index];
    fftIn[index][1] = 0.0;
  }
  
  //move to fourier space
  fft3D(fftIn, fftOut);

  //fftw_execute(p_forward_f); //goes into the convolution
  ifft3D(fftOut,finv);

  for(l=0;l<M_r;l++) {//sum of the quadratures
    for(m=0;m<M_th;m++) 
      for(n=0;n<M_ph;n++) {
	index_quad = n + M_ph*(m + M_th*l);
	#pragma omp parallel for private(index)
	for(index=0;index<N*N*N;index++) {//the xi index
	  
	  F[index][0] = fftOut[index][0] * alphaWtRe[index_quad][index] - fftOut[index][1]*alphaWtIm[index_quad][index];
	  F[index][1] = fftOut[index][0] * alphaWtIm[index_quad][index] + fftOut[index][1]*alphaWtRe[index_quad][index];
	  
	  //for timing purposes only, large N
	  //F[index][0] = fftOut[index][0] * 0.0 - fftOut[index][1]*0.0;
	  //F[index][1] = fftOut[index][0] * 0.0 + fftOut[index][1]*0.0;
	}
	
	
	/*
	//N^2 Convolution for checking
	int ll,mm,nn,x,y,z;
	int n2,n3;
	if(N % 2 == 0) {
	  n2 = N/2;
	  n3 = 1;
	}
	else {
	  n2 = (N-1)/2;
	  n3 = 0;
	}


	for(i=0;i<N;i++) 
	  for(j=0;j<N;j++)
	    for(k=0;k<N;k++) {
	      Ginv[k + N*(j + N*i)][0] = 0.0;
	      Ginv[k + N*(j + N*i)][1] = 0.0;

	      for(ll=0;ll<N;ll++) {
		x = i + n2 - ll;
		if (x < 0)
		  x = N + x;
		else if (x > N-1)
		  x = x - N;
		
		for(mm=0;mm<N;mm++) {
		  y = j + n2 - mm;
		  
		  if (y < 0)
		    y = N + y;
		  else if (y > N-1)
		    y = y - N;
		  
		  for(nn=0;nn<N;nn++) { 
		    z = k + n2 - nn;
		    
		    if (z < 0)
		      z = N + z;
		    else if (z > N-1)
		      z = z - N;
		    
		    Ginv[k + N*(j + N*i)][0] += (F[nn + N*(mm + N*ll)][0]*fftOut[z + N*(y + N*x)][0] - F[nn + N*(mm + N*ll)][1]*fftOut[z + N*(y + N*x)][1]);	
		    Ginv[k + N*(j + N*i)][1] += (F[nn + N*(mm + N*ll)][0]*fftOut[z + N*(y + N*x)][1] + F[nn + N*(mm + N*ll)][1]*fftOut[z + N*(y + N*x)][0]);
		  }
		}
	      }
	    }
	    		  
	*/

	//N log N convolution
		    
	//ifft
	ifft3D(F,Finv);
	//fftw_execute(p_forward_F);
	//fftw_execute(p_forward_f);

#pragma omp parallel for private(index)
	for(index=0;index<N*N*N;index++) {
	  Ginv[index][0] = (Finv[index][0]*finv[index][0] - Finv[index][1]*finv[index][1]);
	  Ginv[index][1] = (Finv[index][1]*finv[index][0] + Finv[index][0]*finv[index][1]);
	}

	//fft
	fft3D(Ginv,G);
	//fftw_execute(p_backward_G);
	
	//prefac2 = scale3*wtR[l]*wtTh[m]*wtPh[n]*pow(r[l],lambda+2)*sin(theta[m])*deta*deta*deta;

	prefac2 = wtR[l]*wtTh[m]*wtPh[n]*pow(r[l],lambda+2)*sin(theta[m]);

#pragma omp parallel for private(index)	
	for(index=0;index<N*N*N;index++) {
	  qHat[index][0] += prefac2*( betaWtRe[index_quad][index] * G[index][0] -  betaWtIm[index_quad][index] * G[index][1]);
	  qHat[index][1] += prefac2*( betaWtIm[index_quad][index] * G[index][0] +  betaWtRe[index_quad][index] * G[index][1]);
	  
	  //Timing check only
	  //qHat[index][0] += prefac2*( 0.0 * G[index][0] -  0.0 * G[index][1]);
	  //qHat[index][1] += prefac2*( 0.0 * G[index][0] +  0.0 * G[index][1]);

	}	
      }
  }

  ifft3D(qHat, fftOut);
 
  //double qmax = 0.0;
  //set Collision output
  for(index=0;index<N*N*N;index++) 
    Q[index] = fftOut[index][0];

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
  double sum;
  double scale = scale3*dv*dv*dv;
  
  //shift the 'v' terms in the exponential to reflect our velocity domain
#pragma omp parallel for private(index,sum,i,j,k)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)	{	
	  index = k + N*(j + N*i);
	  sum = ((double)i + (double)j + (double)k)*L_eta*dv;
	  
	  //dv correspond to the velocity space scaling - ensures that the FFT is properly scaled since fftw does no scaling at all
	  temp[index][0] = scale*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][0] - sin(sum)*in[index][1]);
	  temp[index][1] = scale*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][1] + sin(sum)*in[index][0]);
      }
  //computes fft
  fftw_execute(p_forward);
  
  //shifts the 'eta' terms to reflect our fourier domain
#pragma omp parallel for private(index,sum,i,j,k)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++) 	{
	  index = k + N*(j + N*i);
	  sum = L_v*(eta[i] + eta[j] + eta[k]);
	  
	  out[index][0] = ( cos(sum)*temp[index][0] - sin(sum)*temp[index][1]);
	  out[index][1] = ( cos(sum)*temp[index][1] + sin(sum)*temp[index][0]);
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
  double sum, numScale = scale3;
  double eta3 = deta*deta*deta;

  //shifts the 'eta' terms to reflect our fourier domain
#pragma omp parallel for private(index,sum,i,j,k)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)	
	{
	  index = k + N*(j + N*i);
	  sum = -( ( (double)i + (double)j + (double)k )*L_v*deta );
	  
	  //deta ensures FFT is scaled correctly, since fftw does no scaling at all
	  temp[index][0] = eta3*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][0] - sin(sum)*in[index][1]);
	  temp[index][1] = eta3*wtN[i]*wtN[j]*wtN[k]*(cos(sum)*in[index][1] + sin(sum)*in[index][0]);
      }
  //compute IFFT
  fftw_execute(p_backward);

  //shifts the 'v' terms to reflect our velocity domain
#pragma omp parallel for private(index,sum,i,j,k)
  for(i=0;i<N;i++)
    for(j=0;j<N;j++)
      for(k=0;k<N;k++)	
	{
	  index = k + N*(j + N*i);
	  sum = -( L_eta*(v[i] + v[j] + v[k])  );
	  
	  out[index][0] = (cos(sum)*temp[index][0] - sin(sum)*temp[index][1])*numScale;
	  out[index][1] = (cos(sum)*temp[index][1] + sin(sum)*temp[index][0])*numScale;
      }
  
}

