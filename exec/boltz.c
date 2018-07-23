#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fftw3.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>

#include "initializer.h"
#include "input.h"
#include "output.h"
#include "mesh_setup.h"
#include "restart.h"

#include "collisions.h"
#include "conserve.h"
#include "transportroutines.h"
#include "species.h"

#include "mpi_routines.h"

#include "weights.h"
//#include "pot_weights.h"
#include "aniso_weights.h"

int main(int argc, char **argv) {
    //**********************
    //MPI stuff
    //**********************
    int rank;
    int numNodes;
    int nX_Node;
    MPI_Comm worker;
	int lower;
	int upper;
	int range;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Top-level parameters, read from input
    int N;
    int N3;
    double L_v;
    double Kn;
    double lambda;
    double dt;
    int nT;
    int order;
    int dataFreq;
    int restart;
    double restart_time;
    int initFlag;
    int bcFlag;
    int homogFlag;
    int weightFlag;
    int isoFlag;
    char *meshFile = malloc(80 * sizeof(char));
    int weightgen = 0; // 0 = generate weights before, 1 = on the fly

    //parameters for the spatial mesh, if doing inhomogeneous
    int nX;
    double *x, *dx, dx_min;

    //other top level parameters
    double *v;
    double *zeta;
    double **f_hom;     //levels - species, v-position
    double **f_hom1;    //levels - species, v-position
    double ***f_inhom;  //levels - species, x-postion, v-position
    double ***f_conv;   //levels - species, x-postion, v-position (for intermediate step in splitting)
    double ***f_1;      //levels - species, x-postion, v-position (for intermediate step in 2nd order time integrator)
    double **Q;         //array of collision operator results - levels - species, v-position
    double ***conv_weights; //array of convolution weights - levels - interaction index, zeta, xi
	fftw_complex **qHat; // for fftw 
	fftw_complex **fftw_out;
    double ***qHat_mpi; // for mpi communication (not sure how to send fftw_complex)

    //Species data
    int num_species;
    species *mixture;
    char **species_names;

    //timing stuff for restart
    double totTime, totTime_start;
    double writeTime, writeTime_start;
    int restart_flag;

    //command line arguments
    char *inputFilename = malloc(80 * sizeof(char));
    strcpy(inputFilename, argv[1]);

    char *outputChoices = malloc(80 * sizeof(char));
    strcpy(outputChoices, argv[2]);

    //Variables for main function
    int t;
    int i, j, k, l, m, n;
    int outputCount;

    double t1, t2;

    if ((restart_time > 0) && (rank == 0)) {
        totTime_start = MPI_Wtime();
        totTime = 0;
    }

    //load data from input file
    read_input(&N, &L_v, &Kn, &lambda, &dt, &nT, &order, &dataFreq, &restart, &restart_time, &initFlag, &bcFlag, &homogFlag, &weightFlag, &isoFlag, &meshFile, &num_species, &species_names, inputFilename);

    if (rank) {
        MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &worker);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &worker);
    }

    N3 = N * N * N;
         //////////////////////////////////
         //SETUP                         //
         //////////////////////////////////

         //Allocating and initializing

         load_and_allocate_spec(&mixture, num_species, species_names);

    if (homogFlag == 1) {    //load mesh data
        //if (rank == 0) {
			make_mesh(&nX, &nX_Node, &dx_min, &x, &dx, order, meshFile);
        	printf("Loading mesh\n");
		//}
    }
	
	double *fbuffer = malloc(sizeof(double)*num_species*(nX_Node+(2*order))*N3);
	double *qbuffer = malloc(sizeof(double)*num_species*num_species*N3*2);

    if (rank == 0) printf("Initializing variables %d\n", homogFlag);

    // hector this should probably only be done by rank 0
    if (homogFlag == 0) { // homogeneous case
        allocate_hom(N, &v, &zeta, &f_hom, &f_hom1, &Q, num_species);
        initialize_hom(N, L_v, v, zeta, f_hom, initFlag, mixture);
        t = 0;
    }
    else { // inhomogeneous case
		qHat = (fftw_complex **)malloc(num_species*num_species*sizeof(fftw_complex*));
		fftw_out = (fftw_complex **)malloc(num_species*num_species*sizeof(fftw_complex*));
		for(int index_s = 0; index_s < num_species*num_species; index_s += 1){
			qHat[index_s] = (fftw_complex *)malloc(N3*sizeof(fftw_complex));
			fftw_out[index_s] = (fftw_complex *)malloc(N3*sizeof(fftw_complex));
		}

		qHat_mpi = (double*** )malloc(num_species*num_species*sizeof(double**));
		for(int index_s = 0; index_s < num_species*num_species; index_s += 1){
			qHat_mpi[index_s] = (double**)malloc(N3*sizeof(double*));
			for(int index_n = 0; index_n < N3; index_n += 1){
				qHat_mpi[index_s][index_n] = (double*)malloc(2*sizeof(double));
			}
		}

        if (rank == 0) {
            allocate_inhom(N, nX_Node + (2 * order), &v, &zeta, &f_inhom, &f_conv, &f_1, &Q, num_species);
            initialize_inhom(N, num_species, L_v, v, zeta, f_inhom, f_conv, f_1, mixture, initFlag, nX_Node, x, dx, dt, &t, order, restart, inputFilename);
        } else {
            allocate_inhom(N, nX_Node + (2 * order), &v, &zeta,&f_inhom, &f_conv, &f_1, &Q, num_species);
            initialize_inhom_mpi(N, num_species, L_v, v, zeta, f_inhom, f_conv, f_1, mixture, initFlag, nX_Node, x, dx, dt, &t, order, restart, inputFilename);
        }
    }

//Setup output
		// hector all ranks are doing this
        if (homogFlag == 0)
			initialize_output_hom(N, L_v, restart, inputFilename, outputChoices, mixture, v, num_species);
        else
			initialize_output_inhom(N, L_v, nX, nX_Node, x, dx, restart, inputFilename, outputChoices, mixture, num_species);

//Setup weights

    if ((weightgen == 0 && rank != 0) || homogFlag == 0) {
        printf("Preparing for precomputation of weights...\n");
		if(homogFlag == 1){
        getBounds(&lower, &upper, N, &worker);
		range = upper - lower + 1;
        alloc_weights(range, &conv_weights, num_species * num_species);
		} else {
		lower = 0;
		range = N*N*N;
		alloc_weights(N*N*N, &conv_weights, num_species*num_species);
		}

        if (isoFlag == 1) { //load AnIso weights generated by the AnIso weight generator (seperate program)
            initialize_weights_AnIso(N, zeta, L_v, lambda, weightFlag, conv_weights[0], 0.0);
        }
        else {
            for (i = 0; i < num_species; i++)
                for (j = 0; j < num_species; j++)
                    initialize_weights(lower, range, N, zeta, L_v, lambda, weightFlag, isoFlag, conv_weights[j * num_species + i], mixture[i], mixture[j]);
        }
    }
    else {
        printf("Not precomputing weights; The weights will be computed on the fly...\n");
    }
    outputCount = 0;

//Set up conservation routines

    initialize_conservation(N, v[1] - v[0], v, mixture, num_species);

//////////////////////////////////////////////////
//ALL SETUP COMPLETE                            //
//////////////////////////////////////////////////

    printf("Done with all setup, starting main loop RANK %d\n",rank);
    fflush(stdout);

    writeTime_start = MPI_Wtime();
    totTime_start = MPI_Wtime();

//////////////////////////////////////////////
//SPACE HOMOGENEOUS CASE                    //
///////////////////////////////////////////////
    if(homogFlag == 0) {
		write_streams(f_hom,0);

        while (t < nT) {
            if(rank == 0) printf("In step %d of %d\n", t + 1, nT);
            t1 = omp_get_wtime();

            for (l = 0; l < num_species; l++) {
                for (m = 0; m < num_species; m++) {
                    //ComputeQ(f_hom[l],f_hom[m],Q[m*num_species + l],conv_weights[m*num_species + l]);
                    ComputeQ_maxPreserve(f_hom[l], f_hom[m], Q[m * num_species + l], conv_weights[m * num_species + l]);
                }
            }

            //conserve
            conserveAllMoments(Q);

            t2 = omp_get_wtime();
            //printf("Time elapsed: %g\n", t2 - t1);

            if (order == 1) {
                //update
                for (i = 0; i < N3; i++)
                    for (l = 0; l < num_species; l++)
                        for (m = 0; m < num_species; m++)
                            f_hom[l][i] += dt * Q[m * num_species + l][i] / Kn;
            }
            else if (order == 2) {
                //update
                for (i = 0; i < N3; i++)
                    for (l = 0; l < num_species; l++) {
                        f_hom1[l][i] = f_hom[l][i];
                        for (m = 0; m < num_species; m++)
                            f_hom1[l][i] +=  dt * Q[m * num_species + l][i] / Kn;
                    }

                //compute new collision ops
                for (l = 0; l < num_species; l++) {
                    for (m = 0; m < num_species; m++) {
                        //ComputeQ(f_hom1[l],f_hom1[m],Q[m*num_species + l],conv_weights[m*num_species + l]);
                        ComputeQ_maxPreserve(f_hom1[l], f_hom1[m], Q[m * num_species + l], conv_weights[m * num_species + l]);
                    }
                }

                //conserve
                conserveAllMoments(Q);

                //update 2
                for (i = 0; i < N3; i++)
                    for (l = 0; l < num_species; l++) {
                        f_hom[l][i] = 0.5 * (f_hom[l][i] + f_hom1[l][i]);
                        for (m = 0; m < num_species; m++)
                            f_hom[l][i] += 0.5 * dt * Q[m * num_species + l][i] / Kn;
                    }
            }

            outputCount++;
            if (outputCount % dataFreq == 0) {
				write_streams(f_hom,dt*(t+1));
                outputCount = 0;
            }
            t = t + 1;
        }
    }
/////////////////////////////////////////////
//SPACE INHOMOGENEOUS CASE                 //
/////////////////////////////////////////////
    else {
        if (!restart && rank == 0) {
			write_streams_inhom(f_inhom,0,order);
		}

        if ((rank == 0) && (restart_time > 0)) {
            totTime = MPI_Wtime() - totTime_start;
            writeTime_start = MPI_Wtime();
            //printf("%g\n",totTime);
        }
        while (t < nT) {
            if(rank == 0) printf("In step %d of %d\n", t + 1, nT);
			
			double sum_check = 0;
			for(int sum_check_x = 0; sum_check_x < num_species; sum_check_x += 1) {
			for(int sum_check_y = 0; sum_check_y < (nX_Node) + (2*order); sum_check_y += 1) {
			for(int sum_check_z = 0; sum_check_z < N3; sum_check_z += 1) {
					sum_check += f_inhom[sum_check_x][sum_check_y][sum_check_z];
			}
			}
			}
				
			printf("SUM BEFORE ADVECTION STEP RANK %d (F_INHOM) %f\n", rank, sum_check);
			fflush(stdout);
				 
            ///////////////////////////
            //ADVECTION STEP         //
            ///////////////////////////
            // rank 0
            if (rank == 0) {
                for (m = 0; m < num_species; m++) {
                    if (order == 1) advectOne(f_inhom[m], f_conv[m], m);
                    else advectTwo(f_inhom[m], f_conv[m], m);
                }
            }

			sum_check = 0;
			for(int sum_check_x = 0; sum_check_x < num_species; sum_check_x += 1) {
			for(int sum_check_y = 0; sum_check_y < (nX_Node) + (2*order); sum_check_y += 1) {
			for(int sum_check_z = 0; sum_check_z < N3; sum_check_z += 1) {
					sum_check += f_conv[sum_check_x][sum_check_y][sum_check_z];
			}
			}
			}

			printf("SUM AFTER ADVECTION STEP RANK %d (F_CONV) %f\n", rank, sum_check);
			fflush(stdout);
			
			if(rank == 0) {	
				fcopy(fbuffer,f_conv,num_species,(nX_Node+(2*order)),N3, -1);
			sum_check = 0;
			for(int sum_check_i = 0; sum_check_i < num_species*((nX_Node)+(2*order))*N3; sum_check_i += 1){
				sum_check += fbuffer[sum_check_i];				
			}

			printf("SUM AFTER COPYT TO BUFFER STEP RANK %d (fbuffer) %f\n", rank, sum_check);
			fflush(stdout);
			}

            // broadcast f to all other ranks from rank 0
            int error = MPI_Bcast(fbuffer, num_species * N3 * (nX_Node + (2 * order)), MPI_DOUBLE, 0, MPI_COMM_WORLD);

			if(rank != 0){
				fcopy(fbuffer,f_conv,num_species,(nX_Node+(2*order)),N3, 1);				
			sum_check = 0;
			for(int sum_check_i = 0; sum_check_i < num_species*((nX_Node)+(2*order))*N3; sum_check_i += 1){
				sum_check += fbuffer[sum_check_i];				
			}

			printf("SUM AFTER BROADCAST STEP RANK %d (fbuffer) %f\n", rank, sum_check);
			fflush(stdout);
			}		

            t1 = (double) clock() / (double) CLOCKS_PER_SEC;

            ////////////////////////
            //COLLISION STEP      //
            //////////////////////// 

            for (l = order; l < (nX_Node + order); l++) {

                if (rank != 0) {
                    for (m = 0; m < num_species; m++) {
                        for (n = 0; n < num_species; n++) {
                            ComputeQ_mpi(f_conv[m][l], f_conv[n][l], qHat_mpi[n * num_species + m], conv_weights[n * num_species + m], lower, range);
                        }
                    }
                }

                if (rank == 0) {
                    resetQ(qHat_mpi, num_species,N);
                }

			    qcopy(qbuffer,qHat_mpi,num_species*num_species,N3,-1);

				MPI_Allreduce(MPI_IN_PLACE, qbuffer, num_species * num_species * N3 * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

				qcopy(qbuffer,qHat_mpi,num_species*num_species,N3,1);

				for(int s_ = 0; s_ < num_species*num_species; s_ += 1){
					for(int n_ = 0; n_ < N3; n_ += 1){
						qHat[s_][n_][0] = qHat_mpi[s_][n_][0];
						qHat[s_][n_][1] = qHat_mpi[s_][n_][1];
					}
				}
				
				for(int s_ = 0; s_ < num_species*num_species; s_ += 1) {
					fft3D(qHat[s_], fftw_out[s_], 1);
					for(int n_ = 0; n_ < N3; n_ += 1) {
						Q[s_][n_] = fftw_out[s_][n_][0];
					}
				}	
			if(rank == 0 && l == order){
			sum_check = 0;
			for(int sum_check_x = 0; sum_check_x < num_species*num_species; sum_check_x += 1) {
			for(int sum_check_z = 0; sum_check_z < N3; sum_check_z += 1) {
					sum_check += Q[sum_check_x][sum_check_z];
			}
			}
			printf("RANK %d Q after compute Q %f \n",rank,sum_check);
			fflush(stdout);}

                conserveAllMoments(Q);
			if(rank == 0 && l == order){
			sum_check = 0;
			for(int sum_check_x = 0; sum_check_x < num_species*num_species; sum_check_x += 1) {
			for(int sum_check_z = 0; sum_check_z < N3; sum_check_z += 1) {
					sum_check += Q[sum_check_x][sum_check_z];
			}
			}
			printf("RANK %d Q after conserve %f \n",rank,sum_check);
			fflush(stdout);}



                t2 = (double)clock() / (double)CLOCKS_PER_SEC;
                //if(rank == 0) printf("Time elapsed: %g\n", t2 - t1);
				
				// at the moment this will be done by all the ranks
                for (m = 0; m < num_species; m++) {
                    if (order == 1)
                        for (i = 0; i < N; i++)
                            for (j = 0; j < N; j++)
                                for (k = 0; k < N; k++)
                                    f_inhom[m][l][k + N * (j + N * i)] = f_conv[m][l][k + N * (j + N * i)];
                    else
                        for (i = 0; i < N; i++)
                            for (j = 0; j < N; j++)
                                for (k = 0; k < N; k++)
                                    f_1[m][l][k + N * (j + N * i)] = f_conv[m][l][k + N * (j + N * i)];

                    for (n = 0; n < num_species; n++)
                        for (i = 0; i < N; i++)
                            for (j = 0; j < N; j++)
                                for (k = 0; k < N; k++) {
                                    if (order == 1)
                                        f_inhom[m][l][k + N * (j + N * i)] +=  dt * Q[n * num_species + m][k + N * (j + N * i)] / Kn;
                                    else
                                        f_1[m][l][k + N * (j + N * i)] +=  dt * Q[n * num_species + m][k + N * (j + N * i)] / Kn;
                                }
                }

                // no broadcast of fs because they have already been computed by each rank

				//The RK2 Part
                if (order == 2) {
					printf("RK2 PART\n");
					fflush(stdout);
					if(rank != 0) {
					for (m = 0; m < num_species; m++)
                        for (n = 0; n < num_species; n++)
                            ComputeQ_mpi(f_1[m][l], f_1[n][l], qHat_mpi[n * num_species + m], conv_weights[n * num_species + m], lower, range);
					}
					
					if(rank == 0){
						resetQ(qHat_mpi, num_species, N);
					}

					qcopy(qbuffer,qHat_mpi,num_species*num_species,N3,-1);

					// reduce only to rank 0 since there are no more computeQ
					MPI_Reduce(MPI_IN_PLACE,qbuffer, num_species * num_species * N3*2, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
					
					if(rank == 0) {
					
					qcopy(qbuffer,qHat_mpi,num_species*num_species,N3,-1);
				
					for(int s_ = 0; s_ < num_species*num_species; s_ += 1){
						for(int n_ = 0; n_ < N3; n_ += 1){
							qHat[s_][n_][0] = qHat_mpi[s_][n_][0];
							qHat[s_][n_][0] = qHat_mpi[s_][n_][1];
						}
					}
				
					for(int s_ = 0; s_ < num_species*num_species; s_ += 1) {
						fft3D(qHat[s_], fftw_out[s_], 1);
						for(int n_ = 0; n_ < N3; n_ += 1) {
							Q[s_][n_] = fftw_out[s_][n_][0];
						}
					}	
                    
					conserveAllMoments(Q);

                    for (m = 0; m < num_species; m++) {
                        for (i = 0; i < N; i++)
                            for (j = 0; j < N; j++)
                                for (k = 0; k < N; k++)
                                    f_conv[m][l][k + N * (j + N * i)] = 0.5 * f_conv[m][l][k + N * (j + N * i)] + 0.5 * f_1[m][l][k + N * (j + N * i)];

                        for (n = 0; n < num_species; n++)
                            for (i = 0; i < N; i++)
                                for (j = 0; j < N; j++)
                                    for (k = 0; k < N; k++)
                                        f_conv[m][l][k + N * (j + N * i)] += 0.5 * dt * Q[n * num_species + m][k + N * (j + N * i)] / Kn;
                    }
					}
                }
            }

            ////////////////////////////////
            //ADVECTION STEP (IF NEEDED)  //
            ////////////////////////////////
            if (order == 2 && rank == 0)
                for (m = 0; m < num_species; m++)
                    advectTwo(f_conv[m], f_inhom[m], m);
            
			////////////////////////////////
            //RECORD DATA                 //
            ////////////////////////////////
            outputCount++;
            if (outputCount % dataFreq == 0) {/*
                if (restart_time > 0) {
                    if ((rank == 0) && (restart_time > 0)) {
                        writeTime = MPI_Wtime() - writeTime_start;
                        totTime = MPI_Wtime() - totTime_start;
                        writeTime_start = MPI_Wtime();
                        //printf("%g %g\n",totTime,writeTime);
                        if ((totTime + writeTime) > 0.95 * (double)restart_time) {
                            printf("RESTART TIME REACHED - STORING CURRENT DISTRIBUTION DATA\n");
                            restart_flag = 1;
                            MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                            store_restart(f_inhom, t, inputFilename);
                        }
                        else {
                            restart_flag = 0;
                            MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        }
                    }
                    else {
                        MPI_Bcast(&restart_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
                        if (restart_flag == 1)
                            store_restart(f_inhom, t, inputFilename);
                    }
                    if (restart_flag == 1) {
                        MPI_Barrier(MPI_COMM_WORLD);
                        MPI_Finalize();
                        return 0;
                    }
                }*/
				write_streams_inhom(f_inhom,dt*(t+1),order);
                outputCount = 0;
            }
            t = t + 1;
        }
    }


    if (!homogFlag) { //Homogeneous case
        dealloc_hom(v, zeta, f_hom, Q);
    }
    else
        dealloc_inhom(nX_Node, order, v, zeta, f_inhom, f_conv, f_1, Q);

//////////////////////////////////////
//DEALLOCATION AND WRAP UP          //
//////////////////////////////////////
    if (rank == 0)
        printf("Wrapping up\n");

    if (rank == 0) close_streams(homogFlag);

    if (homogFlag && rank == 0) dealloc_trans();

	if(!homogFlag || (homogFlag && rank != 0)){
    for (i = 0; i < num_species; i++)
        dealloc_weights(range, conv_weights[i]);
	}
    free(conv_weights);
	
    dealloc_coll();
    dealloc_conservation();
    MPI_Finalize();

    return 0;
}
