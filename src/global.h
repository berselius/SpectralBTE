//Contains globally defined variables for inhomogeneous Boltzmann code

//trapezoid rule weights
extern double *wtN;

extern int GL;
extern int N;
extern int ORDER;

extern double glance, glancecons;

//velocity and fourier space discretization variables
extern double *v, L_v, *eta, L_eta, h_v, h_eta;

//commonly used constants/terms
extern double scale3, scale, beta, lambda, cLambda;

//variables used in optimization
extern double CCt[5][5], pivot[5];

//spatial + temporal variables
extern double Lx, *dx, *x, tFinal;
extern double dt;
extern int nX, nT, restart;

extern int rank, numNodes;

//Reused parameters
extern double cfl, Ma;
extern int ICChoice;
extern double Kn, Sthl;
extern double *fM_l, *fM_r;
extern int nThreads;

//Boundary condition information
extern double T0, T1, V0, V1; // For HEAT TRANSFER PROBLEM
extern double TWall, VWall;

