#include <math.h>
#include "transportroutines.h"
#include "boundaryConditions.h"
#include "momentRoutines.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "species.h"
//#include "poisson.h"

//Computes the transport term 

static int N;
static int nX;
static double L_v;
static double h_v;
static double dt;
static int ICChoice;
static double TWall;
static double VWall;
static double T0,T1;
static double V0,V1;
static double *x, *dx, *v;
static double *f_l, *f_r, **f_tmp;
static species *mixture;

static void take_upwind(int j, int k, int dir, int l, double Ma, int rank, int numNodes, double *slope, double **f,  double **f_conv);
static void poiseuille_forcing_term(double **f, double **f_conv, double prefactor, int j, int l, int index);
static void no_flux(int sart_i, int end_i, int f_index, double *slope, double **f, double *f_type);
static void fill_ghost_cells(int *rank_address, int *numNodes_address, double **f);

void initialize_transport(int numV, int numX, double lv, double *xnodes, double *dxnodes, double *vel, int IC, double timestep, double TWall_in, species *mix) {
    N = numV;
    nX = numX; //really NX NODE
    L_v = lv;
    x = xnodes;
    dx = dxnodes;
    v = vel;
    ICChoice = IC;
    dt = timestep;

    h_v = 2 * L_v / (N-1);

    int N3 = N * N * N;
    f_l = malloc(N3 * sizeof(double));
    f_r = malloc(N3 * sizeof(double));
    f_tmp = malloc((nX+4) * sizeof(double *));
    int i;
    for(i=0;i<nX+4;i++)
        f_tmp[i] = malloc(N3 * sizeof(double));

    T0 = 1.;
    T1 = 2.;
    V0 = 0.;
    V1 = 0.;
    TWall = TWall_in;
    VWall = 0.;

    x = xnodes;
    dx = dxnodes;

    mixture = mix;

    initializeBC(N, v, mixture);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double min(double in1, double in2)
{
    if(in1 > in2) {
        return in2;
    }
    else {
        return in1;
    }
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double max(double in1, double in2)
{
    if(in1 > in2) {
        return in1;
    }
    else {
        return in2;
    }
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double minmod(double in1, double in2, double in3)
{
    if ((in1 > 0) && (in2 > 0) && (in3 > 0)) {
        return min(min(in1, in2), in3);
    }
    else if ((in1 < 0) && (in2 < 0) && (in3 < 0)) {
        return max(max(in1, in2), in3);
    }
    else {
        return 0;
    }
}


//Computes first order upwind solution
void upwindOne(double **f, double **f_conv, int id) {
    int i,j,k,l;
    double CFL_NUM;
    double Ma;

    int rank, numNodes;
    int N3 = N * N * N;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
    MPI_Status status;
    //int numamt;


    //FILL GHOST CELLS
    if((rank % 2) == 0) { //EVEN NODES SEND FIRST
        if(rank != (numNodes-1))    //SEND TO RIGHT FIRST
            MPI_Send(f[nX], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);

        if(rank != 0) {//RECIEVE FROM LEFT 
            MPI_Recv(f[0], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
            //MPI_Get_count(&status, MPI_DOUBLE, &numamt);
            //printf("%d got %d\n",rank,numamt);
        }
        else {
            if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poiseuille
                setDiffuseReflectionBC(f[1], f[0], T0, V0, 0, id);
            }
            else if(ICChoice == 1) { // Sudden change in wall temperature 
                setDiffuseReflectionBC(f[1], f[0], 2.0*TWall, VWall, 0, id); //only sets the INCOMING velocities of f_conv
            }
            else if (ICChoice != 6) { //assume that the flow repeats outside the domain... (NOTE: SHOULD I EXPLICITLY COPY THIS?)
                f[0] = f[1]; //come back to fix for periodic later
            }
        }

        if(rank != 0) {
            MPI_Send(f[1], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD); //SEND TO LEFT
        }

        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+1], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status); //RECEIVE FROM RIGHT
        }
        else {
            if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poisseuille
                setDiffuseReflectionBC(f[nX], f[nX+1], T1, V1, 1, id); //sets incoming velocities of f_conv
            }
            else if(ICChoice != 6) {
                f[nX+1] = f[nX]; //assume that the flow repeats outside the domain.. - come back to fix this for periodic
            }
        }        
    }
    else { //ODD NODES RECEIVE FIRST
        MPI_Recv(f[0], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left
        
        if(rank != (numNodes-1)) {
            MPI_Send(f[nX], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD); //send to right
        }
        
        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+1], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status); //recieve from right
        }
        else {
            if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poisseuille
                setDiffuseReflectionBC(f[nX], f[nX+1], T1, V1, 1, id); //sets incoming velocities of f_conv
            }
            else if(ICChoice != 6) {
                f[nX+1] = f[nX]; //assume that the flow repeats outside the domain.. //come back to fix this for periodic
            }
        }
            
        MPI_Send(f[1], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD); //all odd nodes can always send to the left
    }
    if(ICChoice == 6) {
        if(numNodes != 1) {
            if(rank == 0) {
                MPI_Send(f[1], N3, MPI_DOUBLE, numNodes-1, 0, MPI_COMM_WORLD);
                MPI_Recv(f[0], N3, MPI_DOUBLE, numNodes-1, 1, MPI_COMM_WORLD,&status);
            }
            if(rank == numNodes-1) {
                MPI_Recv(f[nX+1], N3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Send(f[nX], N3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
        else {
            f[0] = f[nX];
            f[nX+1] = f[1];
        }
            
    }

    //ALL GHOST CELLS SET, COMMUNICATION COMPLETE

    if(ICChoice == 5) {
        printf("Using default value of 1.0 for forcing parameter\n");
        Ma = 1.0;
    }

    /*
    double *dens, *PoisPot;
    
    if(ICChoice == 6) { //find densities
        if(numNodes != 1) {
            printf("multi-node Poisson not implemented\n");
            exit(1);
        }
        else {
            dens        = malloc(nX*sizeof(double));
            PoisPot = malloc(nX*sizeof(double));
            for(i=0;i<nX+2;i++)
        dens[i] = getDensity(f[i],0);
            Poiss1D(dens,nX+2,PoisPot,dx[1]);
            free(dens);
            free(PoisPot);
        }
            
    }
    */
    int index;
    double prefactor;
    #pragma omp parallel for
    for(l=1;l<nX+1;l++) {
        for(i=0;i<N;i++) {
            for(j=0;j<N;j++) {
                #pragma omp simd
                for(k=0;k<N;k++) {    
                    CFL_NUM = dt*v[i]/dx[l];
                    index = k + N * (j + N * i);
                    //the upwinding
                    if(i < N/2) {
                        f_conv[l][index] = (1.0 + CFL_NUM)*f[l][index] - CFL_NUM*f[l+1][index];
                        //printf("%g %g %g %d %d %d \n", f_r[k + N*(j + N*i)], f[l][k + N*(j + N*i)], f_r[k + N*(j + N*i)] - f[l][k + N*(j + N*i)], i, j, k);
                    }
                    else {
                        f_conv[l][index] = (1.0 - CFL_NUM)*f[l][index] + CFL_NUM*f[l-1][index];
                    }
                }
                //Add forcing terms if Poiseuille
                prefactor = -0.5 * Ma * dt / dx[l];
                poiseuille_forcing_term(f, f_conv, prefactor, j, l, index);
                /*    //Add poisson terms
                if(ICChoice == 6) {
                    if(k == 0) 
                        f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - dt*PoisPot[l]*(f[l][k+1 + N*(j + N*i)])/(2*h_v);
                    else if (k == N-1) 
                        f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] + dt*PoisPot[l]*(f[l][k-1 + N*(j + N*i)])/(2*h_v);
                    else
                        f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - dt*PoisPot[l]*(f[l][k+1 + N*(j + N*i)] - f[l][k-1 + N*(j + N*i)])/(2*h_v);
                }
                */
            }
        }
    }
}

//Computes second order upwind solution, with minmod
void upwindTwo(double **f, double **f_conv, int id) {
    int j,k,l;
    double slope[3];
    double Ma;

    int rank, numNodes;

    if(ICChoice == 5) {
        printf("Using default value of 1.0 for forcing parameter\n");
        Ma = 1.0;
    }

    //Fill ghost cells
    fill_ghost_cells(&rank, &numNodes, f);

    //ghost cells filled

    for(l = 2; l < nX+2; l++) {

        //generate wall values - need the slopes at the wall to get 'em 
        if((l == 2) && (rank == 0)) {
            no_flux(0, N/2, 2, slope, f, f_l);
            if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poiseuille
                setDiffuseReflectionBC(f_l, f_l, T0, V0, 0, id);
            }
            else if(ICChoice == 1) {// Sudden change in wall temperature
                setDiffuseReflectionBC(f_l, f_l, 2.0*TWall, VWall, 0, id); //only sets the INCOMING velocities of f_conv
            }
            else { //ensure no flux
                no_flux(N/2, N, 2, slope, f, f_l);
            }
        }
        else if((l == nX+1) && (rank == numNodes-1)) {//on right wall
            no_flux(N/2, N, nX, slope, f, f_r);
            if(ICChoice == 3 || ICChoice == 5) {  // Heat Transfer or Poiseuille 
                setDiffuseReflectionBC(f_r, f_r, T1, V1, l, id); //sets incoming velocities of f_conv
            }
            else { //ensure no flux
                no_flux(0, N/2, nX, slope, f, f_r);
            }
        }
        #pragma omp parallel for
        for(j = 0; j < N; j++) {
            for(k = 0; k < N; k++) { 
                take_upwind(j, k, 0, l, Ma, rank, numNodes, slope, f, f_conv);
                take_upwind(j, k, 1, l, Ma, rank, numNodes, slope, f, f_conv);
            }
        }
    }
}

static void fill_ghost_cells(int *rank_address, int *numNodes_address, double **f) {
    MPI_Comm_rank(MPI_COMM_WORLD, rank_address);
    MPI_Comm_size(MPI_COMM_WORLD, numNodes_address);
    MPI_Status status;

    int i, j, k, index;
    int N3 = N * N * N;
    int rank = *rank_address;
    int numNodes = *numNodes_address;

    //EVEN NODES SEND FIRST
    if((rank % 2) == 0) {
        fflush(stdout);
        //send to right
        if(rank != (numNodes-1))
            MPI_Send(f[nX+1], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);

        //receive from left, or use extrapolation
        if(rank != 0)
            MPI_Recv(f[1], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
        else {
            #pragma omp parallel for
            for(i=0;i<N;i++)
                for(j=0;j<N;j++)
                    #pragma omp simd
                    for(k=0;k<N;k++) {
                        index = k + N * (j + N * i);
                        f[1][index] = 2*f[2][index] - f[3][index];
                    }
        }

        //send second cell to right
        if(rank != (numNodes-1)) {
            MPI_Send(f[nX], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        }

        //receive second from left, or ignore
        if(rank != 0) {
            MPI_Recv(f[0], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status);
        }

        //send to left
        if(rank != 0) {
            MPI_Send(f[2], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
        }

        //receive from right, or use extrapolation
        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+2], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
        }
        else {
            for(i=0;i<N;i++) {
                for(j=0;j<N;j++) {
                    for(k=0;k<N;k++) {
                        index = k + N * (j + N * i);
                        f[nX+2][index] = 2 * f[nX+1][index] - f[nX][index];
                    }
                }
            }
        }

        //send second to left
        if(rank != 0) {
            MPI_Send(f[3], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD);
        }

        //receive second from right, or ignore
        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+3], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
        }

    }
    else { //ODD NODES RECEIVE FIRST
        //receive from left
        MPI_Recv(f[1], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left


        //send to right
        if(rank != (numNodes-1)) {
            MPI_Send(f[nX+1], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        }

        //receive second from left
        MPI_Recv(f[0], N3, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left

        //send second to right
        if(rank != (numNodes-1)) {
            MPI_Send(f[nX], N3, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
        }

        //receive from right, or use extrapolation
        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+2], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
        }
        else {
            for(i=0;i<N;i++) {
                for(j=0;j<N;j++) {
                    for(k=0;k<N;k++) {
                        index = k + N * (j + N * i);
                        f[nX+2][index] = 2 * f[nX+1][index] - f[nX][index];
                    }
                }
            }
        }
            
        //send to left
        MPI_Send(f[2], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD); //all odd nodes can always send to the left

        //receive second from right, or ignore
        if(rank != (numNodes-1)) {
            MPI_Recv(f[nX+3], N3, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &status);
        }
            
        //send second to left
        MPI_Send(f[3], N3, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD); //all odd nodes can always send to the left
    }
}

static void no_flux(int start_i, int end_i, int f_index, double *slope, double **f, double *f_type) {
    int i, j, k, index;
    #pragma omp parallel for
    for(i = start_i; i < end_i; i++) {
        for(j = 0; j < N; j++) {
            #pragma omp simd
            for(k = 0; k < N; k++) {
                index = k + N * (j + N * i);
                slope[1] = minmod((f[f_index+1][index] - f[f_index][index])/(x[f_index+1] - x[f_index]),
                                    (f[f_index+2][index] - f[f_index+1][index])/(x[f_index+2] - x[f_index+1]),
                                    (f[f_index+2][index] - f[nX][index]) /(x[f_index+2] - x[f_index]));
                f_type[index] = f[f_index+1][index] + 0.5 * dx[f_index+1] * slope[1];
            }
        }
    }
}

static void take_upwind(int j, int k, int dir, int l, double Ma, int rank, int numNodes, double *slope, double **f,  double **f_conv) {
    int i, index;
    double prefactor = -0.25 * Ma * dt / h_v;
    double inside = 0.5 * dx[l] * slope[1];
    double CFL_NUM;

    int start, end;
    if (dir == 0) {
        start = N/2;
        end = N;
    }
    else {
        start = 0;
        end = N/2;
    }

    #pragma omp simd
    for (i = start; i < end; i++) {
        index = k + N * (j + N * i);;
        slope[1+dir] = minmod((f[l+dir][index] - f[l-1+dir][index])/(x[l+dir] - x[l-1+dir]),
                                (f[l+1+dir][index] - f[l+dir][index])/(x[l+1+dir] - x[l+dir]), 
                                (f[l+1+dir][index] - f[l-1+dir][index])/(x[l+1+dir] - x[l-1+dir]));
        slope[dir] = minmod((f[l-1+dir][index] - f[l-2+dir][index])/(x[l-1+dir] - x[l-2+dir]),
                                (f[l+dir][index] - f[l-1+dir][index])/(x[l+dir] - x[l-1+dir]), 
                                (f[l+dir][index] - f[l-2+dir][index])/(x[l+dir] - x[l-2+dir]));

        CFL_NUM = 0.5 * dt * v[i] / dx[l];
                    
        f_conv[l][index] = f[l][index] - CFL_NUM * (f[l][index] + 0.5 * dx[l] * slope[1]);

        if(l == 2 && rank == 0 && dir == 0)  { //f_l is the INCOMING distribution from the left wall
            inside += f[l][index] - f_l[index];
        }
        else if (l == nX+1 && rank == numNodes-1) { //f_r is the INCOMING distribution from the right wall
            inside += f_r[index] - f[l][index];
        }
        else {  //the upwinding
            inside += f[l+dir][index] - f[l-1+dir][index] - 0.5*dx[l-1+dir]*slope[2*dir];
        } 

        f_conv[l][index] = f[l][index] - CFL_NUM * inside;
        poiseuille_forcing_term(f, f_conv, prefactor, j, l, index);
    }
}

static void poiseuille_forcing_term(double **f, double **f_conv, double prefactor, int j, int l, int index) {
    if (ICChoice == 5) {
        if (j == 0) {
            f_conv[l][index] += prefactor * f[l][index + N];
        }
        else if(j == N-1) {
            f_conv[l][index] += prefactor * f[l][index - N];
        }
        else {
            f_conv[l][index] += prefactor * (f[l][index + N] - f[l][index - N]);
        }
    }
}


void advectOne(double **f, double **f_conv, int id) {
    upwindOne(f, f_conv, id);
}

void advectTwo(double **f, double **f_conv, int id) {
    int i, j, k, l;

    //first RK step (calculates u(1) = u(0) + dt f(u(0)) )
    upwindTwo(f, f_tmp, id);
    
    //second RK step (calculates u(2) = u(1) + dt f(u(1)) )
    upwindTwo(f_tmp, f_conv, id);
    int index;
    //average the two steps
    #pragma omp parallel for
    for(l = 2; l < nX+2; l++) {
        for(k = 0; k < N; k++) {
            for(j = 0; j < N; j++) {
                #pragma omp simd
                for(i = 0; i < N; i++) {
                    index = k + N * (j + N * i);
                    f_conv[l][index] = 0.5 * (f[l][index] + f_conv[l][index]);
                }
            }
        }
    }
}

void dealloc_trans() {
    int i;
    free(f_l);
    free(f_r);
    for(i=0;i<nX+4;i++) {
        free(f_tmp[i]);
    }
    free(f_tmp);
}
