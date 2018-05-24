#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <string.h>
#include "scatter.h"
#include <stdio.h>

static double L_v;
static int N;
static double *zeta;
static char *potential_name;
static gsl_error_handler_t* default_handler;

//Stuff for scatting angle calcs
static double (*potential)(double r);
static double (*d_potential)(double r);
static const gsl_root_fdfsolver_type *solvertype;

typedef struct  {
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //r cosphi 0.5 zetalen
  double arg8; //0.5 r zetalen sinphi
  double arg9; //cos(r cosphi zetadot)
  gsl_integration_workspace *w_b;
  gsl_function F_b;
  gsl_integration_workspace *w_ph;
  gsl_function F_ph;
} integration_args_pot;


double BornMayer(double r) {
  return exp(-r);
}

double DebyeHuckel_att(double r) {
  return -exp(-r) / r;
}

double d_DebyeHuckel_att(double r) {
  return (1/r + 1/(r*r))*exp(-r);
}

double DebyeHuckel_rep(double r) {
  return exp(-r) / r;
}

double d_DebyeHuckel_rep(double r) {
  return -(1/r + 1/(r*r))*exp(-r);
}


double chi_integrand(double t, void *params) {
  double *dparams = (double *)params;
  double b = dparams[0];
  double beta = dparams[1];
  double xmax = dparams[2];

  double tsq = t*t;

  //return t / sqrt((1. - potential(b/tsq)*invE - tsq*tsq ) );
  if((1 - 2*beta*potential(b/(xmax - tsq)) - (xmax - tsq)*(xmax - tsq) ) < 0)
    return 0;
  else
    return sqrt(tsq/(1 - 2*beta*potential(b/(xmax - tsq)) - (xmax - tsq)*(xmax - tsq) ) );
}

double ddistance_xstar(double x, void *params) {
  double *dparams = (double *)params;
  double beta = dparams[0];

  return (x+1)*exp(x) - beta;
}

double distance_xstar(double x, void *params) {
  double *dparams = (double *)params;
  double beta = dparams[0];

  return x*exp(x) - beta*(x-1);
}

void distance_xstar_fdf(double x, void *params, double *f, double *df) {
  double *dparams = (double *)params;
  double beta = dparams[0];

  *f = x*exp(x) - beta*(x-1);
  *df = (x+1)*exp(x) - beta;
}

double rhostar(double beta) {
  int status;
  double root, soln;
  gsl_function_fdf F_rho;

  gsl_root_fdfsolver *solver;


  solver = gsl_root_fdfsolver_alloc(solvertype);
  F_rho.f   = &distance_xstar;
  F_rho.df  = &ddistance_xstar;
  F_rho.fdf = &distance_xstar_fdf;
  F_rho.params = &beta;

  gsl_root_fdfsolver_set(solver, &F_rho, log(beta));
  status = GSL_CONTINUE;
  int iter = 0;
  while(status != GSL_SUCCESS) {
    gsl_root_fdfsolver_iterate(solver);
    root = gsl_root_fdfsolver_root(solver);

    status = gsl_root_test_residual(GSL_FN_FDF_EVAL_F(&F_rho,root),1e-10);
    
    iter++;
  }

  soln = root*sqrt((root+1)/(root-1));

  gsl_root_fdfsolver_free(solver);

  return soln;
}

double distance_rm(double r, void *params) {
  double b,beta;
  double *dparams = (double *)params;
  b = dparams[0];
  beta = dparams[1];
  double f = r*(1 - (b/r)*(b/r) - 2*beta*potential(r));

  return f;
}

double ddistance_rm(double r, void *params) {
  double b,beta;
  double *dparams = (double *)params;
  b = dparams[0];
  beta = dparams[1];
  
  double df = r*(2*(b/r)*(b/r)/r - 2*beta*d_potential(r)) + (1 - (b/r)*(b/r) - 2*beta*potential(r));


  return df;
}

void distance_rm_fdf(double r, void *params, double *f, double *df) {
  double b,beta;
  double *dparams = (double *)params;
  b = dparams[0];
  beta = dparams[1];

  *f  = r*(1 - (b/r)*(b/r) - 2*beta*potential(r));
  *df = r*(2*(b/r)*(b/r)/r - 2*beta*d_potential(r) ) + (1 - (b/r)*(b/r) - 2*beta*potential(r));  
}

void distance_rm_fdf_2(double r, void *params, double *f, double *df) {
  double b,beta;
  double *dparams = (double *)params;
  b = dparams[0];
  beta = dparams[1];

  *f  = (1 - (b/r)*(b/r) - 2*beta*potential(r));
  *df = (2*(b/r)*(b/r)/r - 2*beta*d_potential(r));// + (1 - (b/r)*(b/r) + 2*beta*exp(-r)/r);  

  fflush(stdout);
}


//Computes the minimum distance during the interaction
double compute_rm(double b, double beta, double rhostar) {
  gsl_function_fdf F_rm;

  gsl_root_fdfsolver *solver;

  double root;

  solver = gsl_root_fdfsolver_alloc(solvertype);
  default_handler = gsl_set_error_handler_off();

  int status, newt_status;

  //set up rootfinding
  double params[2];
  params[0] = b;
  params[1] = beta;
  F_rm.f = &distance_rm;
  F_rm.df = &ddistance_rm;
  if(b < 1)
    F_rm.fdf = &distance_rm_fdf;
  else
    F_rm.fdf = &distance_rm_fdf_2;

  F_rm.params = params;

  double guess;
  guess = b;
    
  double f,df;

  gsl_root_fdfsolver_set(solver, &F_rm, guess);
  root = guess;
  status = GSL_CONTINUE;
  newt_status = 0;
  int iter = 0;
  while((status != GSL_SUCCESS) && (iter < 20) && (newt_status != GSL_EBADFUNC)) {
    f = GSL_FN_FDF_EVAL_F(&F_rm,root);
    df = GSL_FN_FDF_EVAL_DF(&F_rm,root);
    //printf("%g %g %g\n",root,f, df);
    fflush(stdout);
    newt_status = gsl_root_fdfsolver_iterate(solver);

    root = gsl_root_fdfsolver_root(solver);
    status = gsl_root_test_residual(f,1e-8);
    /*    if ((status == GSL_SUCCESS) && (f < 0)) {
      gsl_root_fdfsolver_set(solver, &F_rm, root + 1e-10);      
      status = GSL_CONTINUE;
      }*/
    iter++;
  }
  //failed, try again closer to 0...
  if((iter == 20) || (newt_status == GSL_EBADFUNC)) {
    guess = sqrt(b*b + beta*beta) - beta;
    gsl_root_fdfsolver_set(solver, &F_rm, guess);
    root = guess;
    status = GSL_CONTINUE;
    newt_status = 0;
    iter = 0;
    while((status != GSL_SUCCESS) && (iter < 20) && (newt_status != GSL_EBADFUNC)) {
      f = GSL_FN_FDF_EVAL_F(&F_rm,root);
      df = GSL_FN_FDF_EVAL_DF(&F_rm,root);
      //printf("%g %g %g\n",root,f, df);
      fflush(stdout);
      newt_status = gsl_root_fdfsolver_iterate(solver);
      
      root = gsl_root_fdfsolver_root(solver);
      status = gsl_root_test_residual(f,1e-8);
      /*    if ((status == GSL_SUCCESS) && (f < 0)) {
	    gsl_root_fdfsolver_set(solver, &F_rm, root + 1e-10);      
	    status = GSL_CONTINUE;
	    }*/
      iter++;
    }    
  }
  if((iter == 20) || (newt_status == GSL_EBADFUNC)) {
    printf("Newton solver failed for %le\n",b);
    exit(1);
  }  

  //printf("%g %g %g %g %d\n",b, guess, root,f, iter);
  fflush(stdout);

  gsl_root_fdfsolver_free(solver);

  gsl_set_error_handler(default_handler);


  return root;
}

//This function computes the scattering angle based on impact parameter b and energy E
double compute_chi(double b, double beta, double rhostar) {
  double rm;
  double tmax;
  double params[3];
  double result;
  double error;
  size_t nevals;

  gsl_function F_chi;
  F_chi.function = &chi_integrand;

  gsl_integration_cquad_workspace *w_chi = gsl_integration_cquad_workspace_alloc(100000);

  params[0] = b;
  params[1] = beta;

  //Step one - find rm
  rm = compute_rm(b,beta,rhostar);

  if (rm == 0)
    return 0;

  //printf("%le %le\n",b,rm);

  tmax = sqrt(b/rm);
  params[2] = b/rm;

  F_chi.params = params;

  gsl_integration_cquad(&F_chi, 0., tmax, 1e-8, 1e-8,  w_chi, &result, &error, &nevals);

  gsl_integration_cquad_workspace_free(w_chi);

  return M_PI - 4.0*result;
}

void initialize_scatter() {
  potential = DebyeHuckel_rep;
  d_potential = d_DebyeHuckel_rep;
  solvertype = gsl_root_fdfsolver_newton;
}
