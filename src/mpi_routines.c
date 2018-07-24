#include <stdio.h>
#include "constants.h"
#include "mpi_routines.h"
#include <mpi.h>

void getBounds(int* lower, int* upper, int N, MPI_Comm* worker) {
        int rank, numRanks;
        int blockSize;

        MPI_Comm_rank(*worker, &rank);
        MPI_Comm_size(*worker, &numRanks);

        int N3 = N*N*N;

        // decide how to split up the weights
        if(N3 % numRanks != 0 && (rank < N3 % numRanks)) {
                blockSize = (N3/numRanks) + 1;
        } else {
                blockSize = (N3/numRanks);
        }

        // determine the lower and upper bounds
        *lower = 0;
        for(int i = 0; i < rank; i += 1){
                if(i < N3 % numRanks){
                        *lower += (N3/numRanks) + 1;
                }
                else {
                        *lower += (N3/numRanks);
                }
        }
        *upper = *lower + blockSize;
}

void resetQ(double*** qHat_mpi, int num_species, int N){
		int N3 = N*N*N;
    	for(int y_ = 0; y_ < num_species*num_species; y_ += 1){
    		for(int x_ = 0; x_ < N3; x_ += 1) {
        		qHat_mpi[y_][x_][0] = 0;
        		qHat_mpi[y_][x_][1] = 0;
    		}
    	}	
}

// if direction == 1; buffer ==> f
// if direction == -1; f ==> buffer
void fcopy(double* buffer, double*** f, int x, int y, int z, int direction){
	int one,two;
	int i,j,k;
	int total = x*y*z;
	if(direction == 1){
	for(i = 0; i < x; i+=1){
		one = i*y*z;
		for(j = 0;  j < y; j+= 1){
			two = j*z;
			for(k = 0; k < z; k+=1){
				f[i][j][k] = buffer[one + two + k];	
			}
		}
	}
	}
	else if(direction == -1){
		for(int index = 0; index < total; index+=1){
			i = index/(z*y);
			j = (index - i*z*y)/z;
			k = (index - i*x*y - j*z);
			buffer[index] = f[i][j][k];
		}
	}	
}

void qcopy(double* buffer, double*** q, int x, int y, int direction){
    int one, two;
    int i,j,k;
    int total = x*y*2;

    if(direction == 1){
    for(i = 0; i < x; i+=1){
        one = i*y*2;
        for(j = 0;  j < y; j+= 1){
			two = j*2;
			for( k = 0; k < 2; k += 1){
                q[i][j][k] = buffer[one + two + k];
			}
        }
    }
    }
    else if(direction == -1){
        for(int index = 0; index < total; index+=1){
            i = index/(y*2);
            j = (index - i*y*2)/2;
			k = (index - i*y*2 - j*2);
            buffer[index] = q[i][j][k];
        }
    }
}
