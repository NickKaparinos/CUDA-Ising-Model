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

__global__ void kernelCalcNewLattice(int* oldLattice, int* newLattice, double* w, int n){
	// Each thread will calculate the new magnetic moment of the (i,j) cell
	int i = blockIdx.x;
	int j = threadIdx.x;

	double influence = 0.0;
	for(int k=-2; k<=2; k++){
		for(int l=-2; l<=2; l++){
			int newI = ( i+k+n )%n;				// (newI,newJ) are the coordinates of the moment that will influence the (i,j) new moment
    		int newJ = ( j+l+n )%n;
           	influence += w[ (k+2)*5 + (l+2) ] * oldLattice[ newI*n + newJ ];
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

    kernelCalcNewLattice<<<n,n>>>(dev_oldLattice, dev_newLattice, dev_w, n);
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

