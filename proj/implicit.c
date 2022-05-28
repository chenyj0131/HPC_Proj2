static char help[] = "Assemble the matrix.\n";

#include <stdlib.h>
#include "petscmat.h"
#include <petsc.h>
#include <petscvec.h>
#include <math.h>
#include <petscksp.h>

int main(int argc,char **argv)
{
	MPI_Comm        comm;
        PetscMPIInt     rank;
        PetscInt        n=10, max_its=1000;
        PetscInt        i, rstart, rend, its, M, col[3];
	PetscReal	rou=1.0, c=1.0, k=1.0, dt=0.1, l=1.0, f_to_set;
	PetscScalar	value[3], one=1, zero=0, minus=-1.0;
	Vec             u, f, uold, u_f;
	Mat		A;
	KSP		ksp;
	
	double		err, sum_u_old=0.0, sum_u=1.0, value_to_set;
	
	// The first column is the type of bcs, 1 for set value, 2 for heat flux
	// The firt row represents point at left end
	double bcs[2][2] = {{1,0},{1,0}};

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
	double const pi = PETSC_PI;

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

	MatSetValue(A, 0, 0, one, INSERT_VALUES);
	MatSetValue(A, M-1, M-1, one, INSERT_VALUES);
	
  	/* Assemble the matrix */
  	MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
	
	// MatView(A,PETSC_VIEWER_STDOUT_WORLD);





	// Create vector u. Local dimension is default.
	VecCreate(PETSC_COMM_WORLD,&u);
        VecSetSizes(u,PETSC_DECIDE,M);
        VecSetFromOptions(u);
	
	VecGetOwnershipRange(u,&rstart,&rend);
	for (i=rstart; i<rend; i++){
		value_to_set = exp(i*dx);
		VecSetValue(u,i,value_to_set,INSERT_VALUES);		
	}
	VecAssemblyBegin(u);
        VecAssemblyEnd(u);
	VecView(u,PETSC_VIEWER_STDOUT_WORLD);
	
	
	// Duplicate the pattern of u to other vectors
	VecDuplicate(u,&uold);
        VecDuplicate(u,&u_f);
        VecDuplicate(u,&f);
	
	// Set value for heat supply, f
	VecGetOwnershipRange(f,&rstart,&rend);
	for (i=rstart; i<rend; i++)
	{
		f_to_set = para2 * sin(l*pi*i*dx);
		VecSetValues(f,1,&i,&f_to_set,INSERT_VALUES);
	}
	// VecView(f,PETSC_VIEWER_STDOUT_WORLD);
	
	VecSet(u,0.0);
	VecSet(uold,0.0);
	VecSet(u_f,0.0);


	// Create the linear solver
	KSPCreate(PETSC_COMM_WORLD,&ksp);

		
	// Iteration start
	for (i=0; i<100; i++){
	
		VecCopy(u,uold);	
		VecAXPY(uold, one, f);
		
		// Apply B.C.s	
		if (bcs[0][0] == 1) VecSetValue(uold, 0, bcs[0][1], INSERT_VALUES);
		else{
			MatSetValue(A, 0, 1, minus, INSERT_VALUES);
			value_to_set = -1.0*dx*bcs[0][1]/k;
			VecSetValue(uold, 0, value_to_set, INSERT_VALUES);
		}
		
		if (bcs[1][0] == 1) VecSetValue(uold, M-1, bcs[1][1], INSERT_VALUES);
		else{
                        MatSetValue(A, M-1, M-2, minus, INSERT_VALUES);
                        value_to_set = -1.0*dx*bcs[1][1]/k;
			VecSetValue(uold, M-1, value_to_set, INSERT_VALUES);
                }

		MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        	MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
		VecAssemblyBegin(uold);
		VecAssemblyEnd(uold);
		
		// MatView(A,PETSC_VIEWER_STDOUT_WORLD);
		// VecView(uold,PETSC_VIEWER_STDOUT_WORLD);
		
		KSPSetOperators(ksp,A,A);
		KSPSetTolerances(ksp,1.e-2/M,1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);
		KSPSetFromOptions(ksp);
		KSPSolve(ksp,uold,u);
		
		// VecView(u,PETSC_VIEWER_STDOUT_WORLD);
		VecSum(u, &sum_u);
		err = fabs(sum_u - sum_u_old);
		if ( err < 1.e-8){
			PetscPrintf(comm, "End at %Dth iter, err=%g\n",i+1,err);
			break;
		}
		sum_u_old = sum_u;
		
	}
	
	VecView(u,PETSC_VIEWER_STDOUT_WORLD);

	

	VecDestroy(&u);
	VecDestroy(&uold);
	VecDestroy(&u_f);
	VecDestroy(&f);
	MatDestroy(&A);

        PetscFinalize();
}









