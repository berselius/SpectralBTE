#include <fftw3.h>
#include "poisson.h"

//Stuff to add Poisson solve to advection term
//Assumes periodic boundary conditions

void Poiss1D(double *rho, int N, double *F, double dx) {

  fftw_complex *RHS, *RHS_hat, *phi, *phi_hat;
  fftw_plan p_forward, p_backward;
  int i;

  p_forward  = fftw_plan_dft_1d(N, RHS, RHS_hat, FFTW_FORWARD, FFTW_ESTIMATE);
  p_backward = fftw_plan_dft_1d(N, phi_hat, phi, FFTW_BACKWARD, FFTW_ESTIMATE);

  RHS     = fftw_malloc(N*sizeof(fftw_complex));
  RHS_hat = fftw_malloc(N*sizeof(fftw_complex));
  phi     = fftw_malloc(N*sizeof(fftw_complex));
  phi_hat = fftw_malloc(N*sizeof(fftw_complex));

  for(i=0;i<N;i++) {
    RHS[i][0] = (rho[i] - 1.0)/N;
    RHS[i][1] = 0.0;
  }
  
  fftw_execute(p_forward);

  for(i=1;i<N;i++) {
    phi_hat[i][0] = -RHS_hat[i][0]/(i*i);
    phi_hat[i][1] = -RHS_hat[i][1]/(i*i);
  }
  phi_hat[0][0] = 0;
  phi_hat[0][1] = 0;

  fftw_execute(p_backward);

  //Now we have phi, just need to return its derivative
  
  for(i=1;i<N-1;i++) {
    F[i] = (phi[i+1][0] - phi[i-1][0])/(2*dx);
  }
  
}
