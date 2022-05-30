static char help[] = "Explicit algorithm using matrix\n";

#include <stdlib.h>
#include "petscmat.h"
#include <petsc.h>
#include <petscvec.h>
#include <math.h>

int main(int argc,char **argv)
{
        MPI_Comm        comm;
        PetscMPIInt     rank;
        PetscInt        n=10, max_its=8000;
        PetscInt        i, rstart, rend, its, M, col[3];
        PetscReal       rou=1.0, c=1.0, k=1.0, dt=0.001, l=1.0, value_to_set, norm_uold=0.0, norm_u=1.0;
	PetscScalar	value[3];
	Vec		u, uold, f;
	Mat		A;

	double		err;
	
	
	PetscInitialize(&argc,&argv,0,help);

        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	
	// n is the number of division
	PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
	PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
        M = n+1;
        double const dx = 1.0/n;
        PetscPrintf(comm, "Matrix size is %d by %d \n", M, M);
        PetscPrintf(comm, "dx=%f\n",dx);
	
	// define consts
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
	if (rstart == 0)	rstart = 1;
	if (rend == M)		rend   = M-1;
	
	value[0] = para1;
	value[1] = 1.0-2.0*para1;
	value[2] = para1;
		
	for (i=rstart; i<rend; i++){
		col[0] = i-1; col[1] = i; col[2] = i+1;
		MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);	
	}
	
	/* Assemble the matrix */
        MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
	MatView(A,PETSC_VIEWER_STDOUT_WORLD);
	
	
	
	/* set value for u */
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

	
	/* Duplicate the pattern of u to other vectors */
	VecDuplicate(u,&uold);
        VecDuplicate(u,&f);
	
	/* Set value for heat supply, f */
	VecGetOwnershipRange(f,&rstart,&rend);
        for (i=rstart; i<rend; i++)
        {
                value_to_set = para2 * sin(l*pi*i*dx);
                VecSetValues(f,1,&i,&value_to_set,INSERT_VALUES);
        }
        VecAssemblyBegin(f);
        VecAssemblyEnd(f);
	VecView(f,PETSC_VIEWER_STDOUT_WORLD);
	
	
	/* iteration start */
	for (its=0; its<max_its; its++){
		VecCopy(u,uold);
		MatMultAdd(A, uold, f, u);
		
		VecNorm(u,NORM_1,&norm_u);
		err = fabs(norm_u - norm_uold);
		
		if ( err < 1.e-8){
                        PetscPrintf(comm, "End at %Dth iter, err=%g\n",its+1,err);
                        break;
                }
		norm_uold = norm_u;
		
	}
	
	VecView(u,PETSC_VIEWER_STDOUT_WORLD);			
	
	
	
	
	
	VecDestroy(&u);
	VecDestroy(&uold);
	VecDestroy(&f);
	MatDestroy(&A);

        PetscFinalize();
}

