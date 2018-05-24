#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <string.h>
#include "pot_weights.h"
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

static double L_v;
static int N;
static double *zeta;
static char *potential_name;
static gsl_error_handler_t* default_handler;

//Stuff for scatting angle calcs
static double (*potential)(double r);
static double (*d_potential)(double r);
static const gsl_root_fdfsolver_type *solvertype;

//Stuff for 'main' weight integration
static int rank;
static int numNodes;

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
double compute_rm(double b, double beta) {
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
    
  double f;

  gsl_root_fdfsolver_set(solver, &F_rm, guess);
  root = guess;
  status = GSL_CONTINUE;
  newt_status = 0;
  int iter = 0;
  while((status != GSL_SUCCESS) && (iter < 20) && (newt_status != GSL_EBADFUNC)) {
    f = GSL_FN_FDF_EVAL_F(&F_rm,root);
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

//This function computes the scattering angle based on impact parameter b and energy E (and assume Debye Len is 1...?)
double compute_chi(double b, double E) {
  double rm;
  double tmax;
  double params[3];
  double result;
  double error;
  size_t nevals;
  double beta = 2/E;

  gsl_function F_chi;
  F_chi.function = &chi_integrand;

  gsl_integration_cquad_workspace *w_chi = gsl_integration_cquad_workspace_alloc(100000);

  params[0] = b;
  params[1] = beta;

  //Step one - find rm
  rm = compute_rm(b,beta);

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

///////////////////////////////////////
// Main weight integration functions //
///////////////////////////////////////

double ghat_b_pot(double b, void *args) {
  integration_args_pot intargs = *((integration_args_pot *)args);

  /*
    Just to remind ourselves...
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //0.5 r cosphi zetalen
  double arg8; //0.5 r sinphi zetalen 
  double arg9; //cos(r cosphi zetadot)
  */

  //printf("In ghat b %g\n",b);

  double r = intargs.arg3;
  double E = 0.5*r*r;
  double chi = compute_chi(b,E);
  
  //return b*(1-cos(chi));

  //printf("%12.6e, %12.6e %12.6e %12.6e\n",b,chi,cos(intargs.arg7*(1-cos(chi)) - intargs.arg6) * j0(intargs.arg8*sin(chi)) - intargs.arg9, chi*chi*(intargs.arg7*0.5*sin(intargs.arg6) - intargs.arg8*intargs.arg0*0.25*intargs.arg9));
  //fflush(stdout);
    

  if(chi > 1e-4)
    return b*( cos(intargs.arg7*(1-cos(chi)) - intargs.arg6) * j0(intargs.arg8*sin(chi)) - intargs.arg9);
  else
    return b*chi*chi*(0.5*intargs.arg7*sin(intargs.arg6) - 0.25*intargs.arg8*intargs.arg8*intargs.arg9);
}


double ghat_phi_pot(double phi, void *args) {
  integration_args_pot intargs = *((integration_args_pot *)args);
  double result,error;

  gsl_function F_b = intargs.F_b;
  
  /*
    Just to remind ourselves...
  double arg0; //zetalen
  double arg1; //xizeta/zetalen
  double arg2; //xiperp
  double arg3; //r
  double arg4; //cosphi
  double arg5; //sinphi
  double arg6; //r cosphi zetadot
  double arg7; //0.5 r cosphi zetalen
  double arg8; //0.5 r sinphi zetalen 
  double arg9; //cos(r cosphi zetadot)
  */

  //printf("In ghat phi %g\n",phi);

  double r = intargs.arg3;

  intargs.arg4 = cos(phi);
  intargs.arg5 = sin(phi);
  intargs.arg6 = r * intargs.arg4 * intargs.arg1;
  intargs.arg7 = 0.5 * r * intargs.arg4 * intargs.arg0;
  intargs.arg8 = 0.5 * r * intargs.arg5 * intargs.arg0;
  intargs.arg9 = cos(r * intargs.arg4 * intargs.arg1);

  F_b.params = &intargs;

  gsl_integration_qagiu(&F_b, 0., 1e-6, 1e-6, 10000, intargs.w_b, &result, &error); 

  //printf("b integral complete %12.6e\n",phi);
  //fflush(stdout);
  
  return intargs.arg5*j0(intargs.arg3*intargs.arg5*intargs.arg2)*result;
}

double ghat_r_pot(double r, void *args) {
  integration_args_pot intargs = *((integration_args_pot *)args);
  double result, error;

  
  printf("In ghat_r, %g\n",r);
  fflush(stdout);

  gsl_function F_ph = intargs.F_ph;

  intargs.arg3 = r;

  F_ph.params = &intargs;
  gsl_integration_qag(&F_ph,0,M_PI,1e-6,1e-6,6,1000,intargs.w_ph,&result,&error);

  return pow(r,3)*result;
}


double gHat3_pot(double zeta1, double zeta2, double zeta3, double xi1, double xi2, double xi3) {
  double result, error;

  gsl_function F_r, F_b, F_ph;

  gsl_integration_workspace *w_r = gsl_integration_workspace_alloc(1000);


  F_r.function = &ghat_r_pot;
  F_b.function = &ghat_b_pot;
  F_ph.function = &ghat_phi_pot;

  printf("Computing weight for %g %g %g %g %g %g\n",zeta1,zeta2,zeta3,xi1,xi2,xi3);

  integration_args_pot intargs;

  double zetalen2 = zeta1*zeta1 + zeta2*zeta2 + zeta3*zeta3;
  double xilen2   = xi1*xi1 + xi2*xi2 + xi3*xi3;
  double xizeta   = xi1*zeta1 + xi2*zeta2 + xi3*zeta3;
  double zetalen  = sqrt(zetalen2);
  double xiperp;

  if( ((xilen2 - xizeta*xizeta/zetalen2) < 0) || (zetalen2 == 0))
    xiperp = 0;
  else
    xiperp = sqrt( xilen2 - xizeta*xizeta/zetalen2);

  intargs.arg0 = zetalen;
  if(zetalen != 0)
    intargs.arg1 = xizeta/zetalen;
  else
    intargs.arg1 = 0.0;
  intargs.arg2 = xiperp;
  intargs.w_b = gsl_integration_workspace_alloc(10000);
  intargs.F_b = F_b;
  intargs.w_ph = gsl_integration_workspace_alloc(1000);
  intargs.F_ph = F_ph;


  F_r.params = &intargs;  

  gsl_integration_qag(&F_r,0,L_v,1e-6,1e-6,6,1000,w_r,&result,&error);

  gsl_integration_workspace_free(w_r);
  gsl_integration_workspace_free(intargs.w_b);
  gsl_integration_workspace_free(intargs.w_ph);

  return 4*M_PI*M_PI*result;
}

//this generates the convolution weights G
void generate_conv_weights_pot(double **conv_weights)
{
  int i, j, k, l, m, n, z;
#pragma omp parallel for private(i,j,k,l,m,n,z) 
  for(z=rank*(N*N*N/numNodes);z<(rank+1)*(N*N*N/numNodes);z++) {
    k = z % N;
    j = ((z-k)/N) % N;
    i = (z - k - N*j)/(N*N);    
    for(l=0;l<N;l++)
      for(m=0;m<N;m++)
	for(n=0;n<N;n++) 
	  conv_weights[k + N*(j + N*i)][n + N*(m + N*l)] =  gHat3_pot(zeta[l], zeta[m], zeta[n], zeta[i], zeta[j], zeta[k]);
  }
}


void initialize_weights_pot(int nodes, double *eta, double Lv, int weightFlag, double **conv_weights, char *pot_name) {
  FILE *fidWeights;
  char buffer_weights[100];
  char output_buffer[100];
  size_t readFlag;

  MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Status status;

  N = nodes;
  zeta = eta;
  L_v = Lv;

  potential_name = pot_name;

  if(strcmp(pot_name,"Born_Mayer") == 0) {
    potential = BornMayer;
    printf("Using Born_Mayer potential\n");
  }
  else if (strcmp(pot_name,"Debye_Huckel_att")==0) {
    potential = DebyeHuckel_att;
    d_potential = d_DebyeHuckel_att;
    printf("Potential: repulsive Debye-Huckel\n");
  }
  else if (strcmp(pot_name,"Debye_Huckel_rep")==0) {
    potential = DebyeHuckel_rep;
    d_potential = d_DebyeHuckel_rep;
    printf("Potential: repulsive Debye-Huckel\n");
  }
  else {
    printf("Potential %s not implemented yet. If you do not wish to use the potential-generated weights please put \"no_pot\" in the potential field of the input file\n",pot_name); 
    exit(0);
  }

  solvertype = gsl_root_fdfsolver_newton;

  int i,j;

  for(i=0;i<N*N*N;i++) {
    conv_weights[i] = malloc(N*N*N*sizeof(double));
  }

  sprintf(buffer_weights,"Weights/N%d_potential_L_v%g_%s.wts",N, L_v,pot_name);
  if(weightFlag == 0) { //Check to see if the weights are there
    if((fidWeights = fopen(buffer_weights,"r"))) {
      printf("Loading weights from file %s\n",buffer_weights);
      for(i=0;i<N*N*N;i++) { 
	readFlag = fread(conv_weights[i],sizeof(double),N*N*N,fidWeights);
	if(readFlag != N*N*N) {
	  printf("Error reading weight file\n");
	  exit(1);
	} 
      }      
    }
    else {
      printf("Stored weights not found for this configuration, generating ...\n");
      generate_conv_weights_pot(conv_weights);
      
      MPI_Barrier(MPI_COMM_WORLD);
      //get weights from everyone else...
      
      if(rank == 0) {
	//dump the weights we've computed into a file
	fidWeights = fopen(buffer_weights,"w");
	for(i=0;i<(N*N*N/numNodes);i++) {
	  fwrite(conv_weights[i],sizeof(double),N*N*N,fidWeights);
	} 
	//receive from all other processes
	for(i=1;i<numNodes;i++) {
	  for(j=0;j<(N*N*N/numNodes);j++) {
	    MPI_Recv(output_buffer,N*N*N,MPI_DOUBLE,i,j + i*N*N*N/numNodes,MPI_COMM_WORLD,&status);
	    fwrite(output_buffer,sizeof(double),N*N*N,fidWeights);
	  }
	}
	if(fflush(fidWeights) != 0) {
	  printf("Something is wrong with storing the weights");
	  exit(0);
	} 
	fclose(fidWeights);
      }
      else {
	for(i=0;i<N*N*N/numNodes;i++)
	  MPI_Send(conv_weights[i],N*N*N,MPI_DOUBLE,0,rank*(N*N*N/numNodes)+i,MPI_COMM_WORLD);
      }
    }
  } 
  else { //weights forced to be regenerated
    printf("Fresh version of weights being computed and stored for this configuration\n");
    generate_conv_weights_pot(conv_weights);

     MPI_Barrier(MPI_COMM_WORLD);
      //get weights from everyone else...
      
      if(rank == 0) {
	//dump the weights we've computed into a file
	fidWeights = fopen(buffer_weights,"w");
	for(i=0;i<(N*N*N/numNodes);i++) {
	  fwrite(conv_weights[i],sizeof(double),N*N*N,fidWeights);
	} 
	//receive from all other processes
	for(i=1;i<numNodes;i++) {
	  for(j=0;j<(N*N*N/numNodes);j++) {
	    MPI_Recv(output_buffer,N*N*N,MPI_DOUBLE,i,j + i*N*N*N/numNodes,MPI_COMM_WORLD,&status);
	    fwrite(output_buffer,sizeof(double),N*N*N,fidWeights);
	  }
	}
	if(fflush(fidWeights) != 0) {
	  printf("Something is wrong with storing the weights");
	  exit(0);
	} 
	fclose(fidWeights);
      }
      else {
	for(i=0;i<N*N*N/numNodes;i++)
	  MPI_Send(conv_weights[i],N*N*N,MPI_DOUBLE,0,rank*(N*N*N/numNodes)+i,MPI_COMM_WORLD);
      }
  }
  printf("Finished with weights\n");
}
