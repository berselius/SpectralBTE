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

void initialize_transport(int numV, int numX, double lv, double *xnodes, double *dxnodes, double *vel, int IC, double timestep, double TWall_in, species *mix) {
  N = numV;
  nX = numX; //really NX NODE
  L_v = lv;
  x = xnodes;
  dx = dxnodes;
  v = vel;
  ICChoice = IC;
  dt = timestep;

  h_v = 2*L_v/(N-1);

  f_l = malloc(N*N*N*sizeof(double));
  f_r = malloc(N*N*N*sizeof(double));
  f_tmp = malloc((nX+4)*sizeof(double *));
  int i;
  for(i=0;i<nX+4;i++)
    f_tmp[i] = malloc(N*N*N*sizeof(double));

  T0 = 1.;
  T1 = 2.;
  V0 = 0.;
  V1 = 0.;
  TWall = TWall_in;
  VWall = 0.;

  x = xnodes;
  dx = dxnodes;

  mixture = mix;

  initializeBC(N,v,mixture);
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double min(double in1, double in2)
{
  if(in1 > in2)
    return in2;
  else
    return in1;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double max(double in1, double in2)
{
  if(in1 > in2)
    return in1;
  else
    return in2;
}

/*$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
double minmod(double in1, double in2, double in3)
{
  if ( (in1 > 0) && (in2 > 0) && (in3  > 0)) {
    return min(min(in1, in2), in3);
  }
  else if ( (in1 < 0) && (in2 < 0) && (in3 < 0)) {
    return max(max(in1, in2), in3);
  }
  else
    return 0;
}


//Computes first order upwind solution
void upwindOne(double **f, double **f_conv, int id) {
  int i,j,k,l;
  double CFL_NUM;
  double Ma;

  int rank, numNodes;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
  MPI_Status status;
  //int numamt;


  //FILL GHOST CELLS
  if((rank % 2) == 0) { //EVEN NODES SEND FIRST
    if(rank != (numNodes-1))  //SEND TO RIGHT FIRST
      MPI_Send(f[nX],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);

    if(rank != 0) {//RECIEVE FROM LEFT 
      MPI_Recv(f[0],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status);
      //MPI_Get_count(&status, MPI_DOUBLE, &numamt);
      //printf("%d got %d\n",rank,numamt);
    }
    else {
      if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poiseuille
	setDiffuseReflectionBC(f[1], f[0], T0, 0, id);
      }
      else if(ICChoice == 1) { // Sudden change in wall temperature	
	setDiffuseReflectionBC(f[1], f[0], 2.0*TWall, 0, id); //only sets the INCOMING velocities of f_conv
      }
      else if (ICChoice != 6) //assume that the flow repeats outside the domain... (NOTE: SHOULD I EXPLICITLY COPY THIS?)
	f[0] = f[1]; //come back to fix for periodic later
    }

    if(rank != 0)
      MPI_Send(f[1],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD); //SEND TO LEFT

    if(rank != (numNodes-1))
      MPI_Recv(f[nX+1],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status); //RECEIVE FROM RIGHT
    else {
      if(ICChoice == 3 || ICChoice == 5) // Heat Transfer or Poisseuille
	setDiffuseReflectionBC(f[nX], f[nX+1], T1, 1, id); //sets incoming velocities of f_conv
      else if(ICChoice != 6)
	f[nX+1] = f[nX]; //assume that the flow repeats outside the domain.. - come back to fix this for periodic
    }    
  }
  else { //ODD NODES RECEIVE FIRST
    MPI_Recv(f[0],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left
    
    if(rank != (numNodes-1))
      MPI_Send(f[nX],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD); //send to right
    
    if(rank != (numNodes-1))
      MPI_Recv(f[nX+1],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status); //recieve from right
    else {
      if(ICChoice == 3 || ICChoice == 5) // Heat Transfer or Poisseuille
	setDiffuseReflectionBC(f[nX], f[nX+1], T1, 1, id); //sets incoming velocities of f_conv
      else if(ICChoice != 6)
	f[nX+1] = f[nX]; //assume that the flow repeats outside the domain.. //come back to fix this for periodic
    }
      
    MPI_Send(f[1],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD); //all odd nodes can always send to the left
  }
  if(ICChoice == 6) {
    if(numNodes != 1) {
      if(rank == 0) {
	MPI_Send(f[1],N*N*N,MPI_DOUBLE,numNodes-1,0,MPI_COMM_WORLD);
	MPI_Recv(f[0],N*N*N,MPI_DOUBLE,numNodes-1,1,MPI_COMM_WORLD,&status);
      }
      if(rank == numNodes-1) {
	MPI_Recv(f[nX+1],N*N*N,MPI_DOUBLE,0,0,MPI_COMM_WORLD,&status);
	MPI_Send(f[nX]  ,N*N*N,MPI_DOUBLE,0,1,MPI_COMM_WORLD);
      }
    }
    else {
      f[0]    = f[nX];
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
      dens    = malloc(nX*sizeof(double));
      PoisPot = malloc(nX*sizeof(double));
      for(i=0;i<nX+2;i++)
	dens[i] = getDensity(f[i],0);
      Poiss1D(dens,nX+2,PoisPot,dx[1]);
      free(dens);
      free(PoisPot);
    }
      
  }
  */

  for(l=1;l<nX+1;l++)  {
    for(i=0;i<N;i++)
      for(j=0;j<N;j++) {
	for(k=0;k<N;k++) {	
	  CFL_NUM = dt*v[i]/dx[l];
	  //the upwinding
	  if(i < N/2) {
	    f_conv[l][k + N*(j + N*i)] = (1.0 + CFL_NUM)*f[l][k + N*(j + N*i)] - CFL_NUM*f[l+1][k + N*(j + N*i)];
	    //printf("%g %g %g %d %d %d \n", f_r[k + N*(j + N*i)], f[l][k + N*(j + N*i)], f_r[k + N*(j + N*i)] - f[l][k + N*(j + N*i)], i, j, k);
	  }
	  else {
	    f_conv[l][k + N*(j + N*i)] = (1.0 - CFL_NUM)*f[l][k + N*(j + N*i)] + CFL_NUM*f[l-1][k + N*(j + N*i)];
	  }
	}
	//Add forcing terms if Poiseuille
	if(ICChoice == 5) {
	  if(j == 0) 
	    f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*dt/(2*dx[l]) * f[l][k + N*((j+1) + N*i)];
	  else if(j == N-1) 
	    f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*dt/(2*dx[l]) * f[l][k + N*((j-1) + N*i)];
	  else
	    f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*dt/(2*dx[l]) * (f[l][k + N*((j+1) + N*i)] - f[l][k + N*((j-1) + N*i)]);
	}
	/*	//Add poisson terms
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

//Computes second order upwind solution, with minmod
void upwindTwo(double **f, double **f_conv, int id) {
  int i,j,k,l;
  double slope[3];
  double CFL_NUM;

  double Ma;

  int rank, numNodes;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numNodes);
  MPI_Status status;

  if(ICChoice == 5) {
    printf("Using default value of 1.0 for forcing parameter\n");
    Ma = 1.0;
  }

  //Fill ghost cells

  //EVEN NODES SEND FIRST
  if((rank % 2) == 0) {
    fflush(stdout);
    //send to right
    if(rank != (numNodes-1))
      MPI_Send(f[nX+1],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);

    //receive from left, or use extrapolation
    if(rank != 0)
      MPI_Recv(f[1],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status);
    else {
      for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	  for(k=0;k<N;k++)
	    f[1][k + N*(j + N*i)] = 2*f[2][k + N*(j + N*i)] - f[3][k + N*(j + N*i)];
    }

    //send second cell to right
    if(rank != (numNodes-1))
      MPI_Send(f[nX],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);

    //receive second from left, or ignore
    if(rank != 0)
      MPI_Recv(f[0],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status);

    //send to left
    if(rank != 0)
      MPI_Send(f[2],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD);

    //receive from right, or use extrapolation
    if(rank != (numNodes-1))
      MPI_Recv(f[nX+2],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status);
    else {
      for(i=0;i<N;i++)
	for(j=0;j<N;j++)
	  for(k=0;k<N;k++)
	    f[nX+2][k + N*(j + N*i)] = 2*f[nX+1][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)];
    }

    //send second to left
    if(rank != 0)
      MPI_Send(f[3],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD);

    //receive second from right, or ignore
    if(rank != (numNodes-1))
      MPI_Recv(f[nX+3],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status);

  }
  else { //ODD NODES RECEIVE FIRST
    //receive from left
    MPI_Recv(f[1],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left


    //send to right
    if(rank != (numNodes-1))
      MPI_Send(f[nX+1],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);


    //receive second from left
    MPI_Recv(f[0],N*N*N,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD, &status); //all odd nodes will always have stuff from the left

    //send second to right
    if(rank != (numNodes-1))
      MPI_Send(f[nX],N*N*N,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);

    //receive from right, or use extrapolation
    if(rank != (numNodes-1))
      MPI_Recv(f[nX+2],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status);
    else {
      for(i=-1;i<N;i++)
	for(j=0;j<N;j++)
	  for(k=0;k<N;k++)
	    f[nX+2][k + N*(j + N*i)] = 2*f[nX+1][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)];
    }
      
    //send to left
    MPI_Send(f[2],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD); //all odd nodes can always send to the left

    //receive second from right, or ignore
    if(rank != (numNodes-1))
      MPI_Recv(f[nX+3],N*N*N,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD, &status);
      
    //send second to left
    MPI_Send(f[3],N*N*N,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD); //all odd nodes can always send to the left
  }

  //ghost cells filled

  for(l=2;l<nX+2;l++) {

    //generate wall values - need the slopes at the wall to get 'em	
    if((l == 2) && (rank == 0)) {
      for(i=0;i<N/2;i++)
	for(j=0;j<N;j++)
	  for(k=0;k<N;k++) {
	    slope[1] = minmod((f[2][k + N*(j + N*i)] - f[1][k + N*(j + N*i)])/(x[2] - x[1]),
			      (f[3][k + N*(j + N*i)] - f[2][k + N*(j + N*i)])/(x[3] - x[2]), 
			      (f[3][k + N*(j + N*i)] - f[1][k + N*(j + N*i)])/(x[3] - x[1]));
	    f_l[k + N*(j + N*i)] = f[2][k + N*(j + N*i)] - 0.5*dx[2]*slope[1];
	  }
      if(ICChoice == 3 || ICChoice == 5) { // Heat Transfer or Poiseuille
	setDiffuseReflectionBC(f_l, f_l, T0, 0, id);
      }
      else if(ICChoice == 1) {// Sudden change in wall temperature
	setDiffuseReflectionBC(f_l, f_l, 2.0*TWall, 0, id); //only sets the INCOMING velocities of f_conv
      }
      else { //ensure no flux
	for(j=0;j<N;j++) 
	  for(k=0;k<N;k++) 
	    for(i=N/2;i<N;i++) {	
	      slope[1] = minmod((f[2][k + N*(j + N*i)] - f[1][k + N*(j + N*i)])/(x[2] - x[1]),
			      (f[3][k + N*(j + N*i)] - f[2][k + N*(j + N*i)])/(x[3] - x[2]), 
			      (f[3][k + N*(j + N*i)] - f[1][k + N*(j + N*i)])/(x[3] - x[1]));
	      f_l[k + N*(j + N*i)] = f[2][k + N*(j + N*i)] + 0.5*dx[2]*slope[1]; //matches the flux leaving cell 0
	    }
	  
	
      }
    }
    
    if((l == nX+1) && (rank == numNodes-1)) {//on right wall
      
      for(i=N/2;i<N;i++)
	for(j=0;j<N;j++)
	  for(k=0;k<N;k++) {
	    slope[1] = minmod((f[nX+1][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)])/(x[nX+1] - x[nX]),
			      (f[nX+2][k + N*(j + N*i)] - f[nX+1][k + N*(j + N*i)])/(x[nX+2] - x[nX+1]), 
			      (f[nX+2][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)]) /(x[nX+2] - x[nX]));
	    f_r[k + N*(j + N*i)] = f[nX+1][k + N*(j + N*i)] + 0.5*dx[nX+1]*slope[1];	      
	  }
      if(ICChoice == 3 || ICChoice == 5) // Heat Transfer or Poiseuille	
	setDiffuseReflectionBC(f_r, f_r, T1, l, id); //sets incoming velocities of f_conv
      else { //ensure no flux
	for(i=0;i<N/2;i++) 
	  for(j=0;j<N;j++) 
	    for(k=0;k<N;k++) {
	      slope[1] = minmod((f[nX+1][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)])/(x[nX+1] - x[nX]),
				(f[nX+2][k + N*(j + N*i)] - f[nX+1][k + N*(j + N*i)])/(x[nX+2] - x[nX+1]), 
				(f[nX+2][k + N*(j + N*i)] - f[nX][k + N*(j + N*i)]) /(x[nX+2] - x[nX]));
	      f_r[k + N*(j + N*i)] = f[nX+1][k + N*(j + N*i)] - 0.5*dx[nX+1]*slope[1]; //matches the flux leaving cell 0
	    }
	  
	
      }
    }
    
    for(j=0;j<N;j++) {
      for(k=0;k<N;k++) { 
	for(i=N/2;i<N;i++) {	
	  //upwind coming from the left
	  //generate the local slopes
	  slope[1] = minmod((f[l][k + N*(j + N*i)] - f[l-1][k + N*(j + N*i)])/(x[l] - x[l-1]),
			    (f[l+1][k + N*(j + N*i)] - f[l][k + N*(j + N*i)])/(x[l+1] - x[l]), 
			    (f[l+1][k + N*(j + N*i)] - f[l-1][k + N*(j + N*i)])/(x[l+1] - x[l-1]));
	  slope[0] = minmod((f[l-1][k + N*(j + N*i)] - f[l-2][k + N*(j + N*i)])/(x[l-1] - x[l-2]),
			    (f[l][k + N*(j + N*i)] - f[l-1][k + N*(j + N*i)])/(x[l] - x[l-1]), 
			    (f[l][k + N*(j + N*i)] - f[l-2][k + N*(j + N*i)])/(x[l] - x[l-2]));
	  
	  
	  CFL_NUM = 0.5*dt*v[i]/dx[l];
	  if( (l == 2) && (rank == 0) )  {
	    //f_l is the INCOMING distribution from the left wall
	    f_conv[l][k + N*(j + N*i)] = f[l][k + N*(j + N*i)] - CFL_NUM*(f[l][k + N*(j + N*i)] + 0.5*dx[l]*slope[1] - f_l[k + N*(j + N*i)]);
	  }
	  else {
	    //the upwinding
	    f_conv[l][k + N*(j + N*i)] = f[l][k + N*(j + N*i)] - CFL_NUM*(f[l][k + N*(j + N*i)] + 0.5*dx[l]*slope[1] - (f[l-1][k + N*(j + N*i)] + 0.5*dx[l-1]*slope[0]));
	  }
	  //Add forcing terms if Poiseuille
	  if(ICChoice == 5) {
	    if(j == 0) 
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * f[l][k + N*((j+1) + N*i)];
	    else if(j == N-1) 
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * f[l][k + N*((j-1) + N*i)];
	    else
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * (f[l][k + N*((j+1) + N*i)] - f[l][k + N*((j-1) + N*i)]);
	  }   
	}
	for(i=0;i<N/2;i++) {
	  //upwind coming from the right
	  //generate the local slopes
	  slope[2] = minmod((f[l+1][k + N*(j + N*i)] - f[l][k + N*(j + N*i)])/(x[l+1] - x[l]),
			    (f[l+2][k + N*(j + N*i)] - f[l+1][k + N*(j + N*i)])/(x[l+2] - x[l+1]), 
			    (f[l+2][k + N*(j + N*i)] - f[l][k + N*(j + N*i)])/(x[l+2] - x[l]));
	  slope[1] = minmod((f[l][k + N*(j + N*i)]   - f[l-1][k + N*(j + N*i)])/(x[l] - x[l-1]),
			    (f[l+1][k + N*(j + N*i)] - f[l][k + N*(j + N*i)])/(x[l+1] - x[l]), 
			    (f[l+1][k + N*(j + N*i)] - f[l-1][k + N*(j + N*i)])/(x[l+1] - x[l-1]));
	  
	  CFL_NUM = 0.5*dt*v[i]/dx[l];
	  if( (l == nX+1) && (rank == numNodes-1) ) {
	    //f_r is the INCOMING distribution from the right wall
	    f_conv[l][k + N*(j + N*i)] = f[l][k + N*(j + N*i)] - CFL_NUM*(f_r[k + N*(j + N*i)] - (f[l][k + N*(j + N*i)] - 0.5*dx[l]*slope[1]));
	  }  
	  else {
	    //the upwinding
	    f_conv[l][k + N*(j + N*i)] = f[l][k + N*(j + N*i)] - CFL_NUM*(f[l+1][k + N*(j + N*i)] - 0.5*dx[l+1]*slope[2] - (f[l][k + N*(j + N*i)] - 0.5*dx[l]*slope[1]));		    
	  }		    
	  //Add forcing terms if Poiseuille
	  if(ICChoice == 5) {
	    if(j == 0) 
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * f[l][k + N*((j+1) + N*i)];
	    else if(j == N-1) 
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * f[l][k + N*((j-1) + N*i)];
	    else
	      f_conv[l][k + N*(j + N*i)] = f_conv[l][k + N*(j + N*i)] - Ma*0.5*dt/(2*h_v) * (f[l][k + N*((j+1) + N*i)] - f[l][k + N*((j-1) + N*i)]);
	  }
	}
      }
    }
  }
}


void advectOne(double **f, double **f_conv, int id) {
  upwindOne(f,f_conv, id);
}

void advectTwo(double **f, double **f_conv, int id) {
  int i,j,k,l;

  //first RK step (calculates u(1) = u(0) + dt f(u(0)) )
  upwindTwo(f,f_tmp,id);
  
  //second RK step (calculates u(2) = u(1) + dt f(u(1)) )
  upwindTwo(f_tmp,f_conv,id);
  
  //average the two steps
  for(l=2;l<(nX+2);l++)
    for(k=0;k<N;k++)
      for(j=0;j<N;j++)
	for(i=0;i<N;i++)
	  f_conv[l][k + N*(j + N*i)] = 0.5*(f[l][k + N*(j + N*i)] + f_conv[l][k + N*(j + N*i)]);
}

void dealloc_trans() {
  int i;
  free(f_l);
  free(f_r);
  for(i=0;i<nX+4;i++)
    free(f_tmp[i]);
  free(f_tmp);
}
