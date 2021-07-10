#include <math.h>
//! Ising model evolution
/*!

  \param G      Spins on the square lattice             [n-by-n]
  \param w      Weight matrix                           [5-by-5]
  \param k      Number of iterations                    [scalar]
  \param n      Number of lattice points per dim        [scalar]

  NOTE: Both matrices G and w are stored in row-major format.
*/

int oldLatticeIndex(int i, int j, int n, int k, int l){
    int newI = ( i+k+n )%n;
    int newJ = ( j+l+n )%n;
    
    return newI*n + newJ;
}

void calcNewLattice(int* oldLattice, int* newLattice, double* w, int n, int iteration){
    if( iteration != 0 ){                   // If its not the first iteration
        for(int i=0; i<n*n; i++){
            oldLattice[i] = newLattice[i];  // oldLattice is now newLattice and we will calculate the newLattice
        }
    }
    
    // Calc newLattice
    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){         // For each point calculate influence
            double influence = 0.0;
            for(int k=-2; k<=2; k++){
                for(int l=-2; l<=2; l++){
                    influence += w[ (k+2)*5 + (l+2) ] * oldLattice[ oldLatticeIndex(i, j, n, k, l) ];
                }
            }
            
            if( fabs(influence) < 1e-6 ){
                newLattice[i*n + j] = oldLattice[i*n + j];
            }else if( influence < 0.0 ){
                newLattice[i*n + j] = -1.0;
            }else{
                newLattice[i*n + j] =  1.0;
            }
        }
    }
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
    }
    
    for(int i=0; i<n*n; i++){
        G[i] = newLattice[i];
    }
    
    free( oldLattice );
    free( newLattice );

}
