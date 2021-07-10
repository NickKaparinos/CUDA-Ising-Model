#include <math.h>
#include <stdlib.h>
//! Ising model evolution
/*!
  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

__global__ void kernelCalcNewLattice(int* oldLattice, int* newLattice, double* w, int n,int dim){
    // Each block will calculate the new magnetic moments of a block of the lattice.The top left cell of that block has coordinates (iStart,jStart).
    // Each thread will calculate a column of that block.
    __shared__ int oldLatticeShared[2916];     // size = (dim+4)^2 

	int iStart = blockIdx.x * blockDim.x;
	int jStart = blockIdx.y * blockDim.x;

    int oldLatticeSharedIndex = 0;              // Used to store values to oldLatticeShared

    //Fill shared memory
    for(int i= iStart-2; i <= iStart+(dim+1); i++){         // Each dim x dim block of moments needs a (dim+4) x (dim+4) block of the oldLattice to calculate the new moments
        for(int j= jStart-2; j <= jStart+(dim+1); j++){     // This (dim+4) x (dim+4) block is stored in oldLatticeShared
            int iPeriodic = ( i+n )%n;
            int jPeriodic = ( j+n )%n;
            oldLatticeShared[ oldLatticeSharedIndex ] = oldLattice[ iPeriodic*n + jPeriodic ];
            oldLatticeSharedIndex++;
        }
    }

    int j = jStart  + threadIdx.x;              // Each thread will compute a different column of moments

    if (j < n){
        for(int i = iStart; i <= (iStart + (dim-1)) && ( i<n ); i++){
            double influence = 0.0;
            for(int k=-2; k<=2; k++){
                for(int l=-2; l<=2; l++){
                    int newI = ( i+k+n )%n;                             // (newI,newJ) are the coordinates of the moment that will influence the (i,j) new moment
                    int newJ = ( j+l+n )%n;                             // (newI,newJ) refer to the old n x n lattice, but everything needed is stored in oldLatticeShared
                    int sharedIIndex = ( newI - (iStart -2) +n )%n;     // (sharedIIndex,sharedJIndex) is where (newI,newJ) is stored
                    int sharedJIndex = ( newJ - (jStart -2) +n )%n;

                    influence += w[ (k+2)*5 + (l+2) ] * oldLatticeShared[ sharedIIndex*(dim+4) + sharedJIndex ];
                }
            }
            if( fabs(influence) < 1e-4 ){
                newLattice[i*n + j] = oldLattice[i*n + j];
            }else if( influence < 0.0 ){
                newLattice[i*n + j] = -1.0;
            }else{
                newLattice[i*n + j] =  1.0;
            }
        }
    }

}

void calcNewLattice(int* oldLattice, int* newLattice, double* w, int n, int iteration){
    if( iteration != 0 ){                   // If its not the first iteration
        for(int i=0; i<n*n; i++){
            oldLattice[i] = newLattice[i];  // oldLattice is now newLattice and we will calculate the newLattice
        }
    }

 	int* dev_oldLattice;
 	int* dev_newLattice;
 	double* dev_w;

 	cudaMallocManaged( &dev_oldLattice, n*n*sizeof(int) );
 	cudaMallocManaged( &dev_newLattice, n*n*sizeof(int) );
 	cudaMallocManaged( &dev_w, 25*sizeof(double) );

 	for(int i=0; i<n*n; i++){
 		dev_oldLattice[i] = oldLattice[i];
 		dev_newLattice[i] = newLattice[i];
 	}

 	for(int i=0; i<25; i++){
 		dev_w[i] = w[i];
 	}

    int dim = 50;
    int numOfBlocks = ceil((double)n/dim);
                                                                // The grid consists of numOfBlocks x numOfBlocks blocks
    dim3 gridSize = dim3( numOfBlocks, numOfBlocks);            // Each block will calculate the moments of a dim x dim block of the lattice
    dim3 blockSize = dim3( dim );                               // Each thread will the calculate the moments of a column of that block

    kernelCalcNewLattice<<<gridSize,blockSize>>>(dev_oldLattice, dev_newLattice, dev_w, n, dim);
    cudaDeviceSynchronize();

    for(int i=0; i<n*n; i++){
 		oldLattice[i] = dev_oldLattice[i];
 		newLattice[i] = dev_newLattice[i];
    }

    cudaFree( dev_oldLattice );
    cudaFree( dev_newLattice );
    cudaFree( dev_w );

}

void ising( int *G, double *w, int k, int n){
    int* oldLattice;
    int* newLattice;

  	oldLattice = (int*) malloc(n*n*sizeof(int));
    newLattice = (int*) malloc(n*n*sizeof(int));



    for(int i=0; i<n*n; i++){
        oldLattice[i] = G[i];
    }

    for(int i=0; i<k; i++){
        if( i!=0 ){             // If it is not the first iteration
            int same = 1;
            for(int j=0; j<n*n; j++){                   // Check if the lattice has changed in the last iteration
                if( oldLattice[j] != newLattice[j]){
                    same = 0;
                    break;
                }
            }

            if( same ){         // If it didnt change, break
                break;
            }
        }
        calcNewLattice(oldLattice, newLattice, w, n, i);
        cudaDeviceSynchronize();
    }

    for(int i=0; i<n*n; i++){
        G[i] = newLattice[i];
    }

    free( oldLattice );
    free( newLattice );

}

