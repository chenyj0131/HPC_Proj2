static char help[] = "Assemble the matrix.\n";

#include <stdlib.h>
#include "petscmat.h"
#include <petsc.h>
#include <petscvec.h>

int main(int argc,char **argv)
{
	MPI_Comm        comm;
        PetscMPIInt     rank;
        PetscInt        n=10, max_its=10000;
        PetscInt        i, rstart, rend, its, M, col[3];
	PetscReal	rou=1.0, c=1.0, k=1.0, dt=1.0;
	PetscScalar	value[3];
	Vec             u, f, uold, u_f;
	Mat		A;



	PetscInitialize(&argc,&argv,0,help);

        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


        // n is the number of division
        PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
        M = n+1;
	double const dx = 1.0/n;
        PetscPrintf(comm, "Matrix size is %d by %d \n", M, M);
	printf("dx = %f\n",dx);
	
	// define to consts
	double const para1 = dt * k / dx / dx / rou / c;
	double const para2 = dt / rou / c;
	
	// Preallocate matrix A
	MatCreate(PETSC_COMM_WORLD,&A);
	MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,M);
	MatSetFromOptions(A);
	MatMPIAIJSetPreallocation(A,3,NULL,2,NULL);
	MatSetUp(A);
	
	// Set values to A
	MatGetOwnershipRange(A,&rstart,&rend);
	PetscPrintf(PETSC_COMM_SELF, "rank [%d] rstart = %d , rend = %d \n\n\n", rank, rstart, rend);

	if (rstart == 0) 
  	{
    	  rstart = 1;
    	  //i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    	  //MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
     	}
  
  	if (rend == M) 
  	{
    	  rend = M-1;
    	  //i = n-1; col[0] = n-2; col[1] = n-1; value[0] = -1.0; value[1] = 2.0;
    	  //MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
  	}

  	/* Set entries corresponding to the mesh interior */
  	value[0] = -1.0*para1; value[1] = 1.0 + 2.0*para1; value[2] = -1.0*para1;
  	for (i=rstart; i<rend; i++) 
  	{
    	  col[0] = i-1; col[1] = i; col[2] = i+1;
    	  MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
  	}
	
	
	
	
	
  	/* Assemble the matrix */
  	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
	
	MatView(A,PETSC_VIEWER_STDOUT_WORLD);













	// Create vector u. Local dimension is default.
	VecCreate(PETSC_COMM_WORLD,&u);
        VecSetSizes(u,PETSC_DECIDE,M);
        VecSetFromOptions(u);

	// Duplicate the pattern of u to other vectors
	VecDuplicate(u,&uold);
        VecDuplicate(u,&u_f);
        VecDuplicate(u,&f);


	VecDestroy(&u);
	VecDestroy(&uold);
	VecDestroy(&u_f);
	VecDestroy(&f);
	MatDestroy(&A);

        PetscFinalize();
}
