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
        PetscInt        n=10, max_its=5000000;
        PetscInt        i, rstart, rend, its, M, col[3], index;
        PetscReal       rou=1.0, c=1.0, k=1.0, dt=0.001, l=1.0, value_to_set;
	PetscScalar	value[3], max_value, minus=-1.0;
	Vec		u, uold, f, diff, u_exact;
	Mat		A;
	

	double		err;
	
	
	PetscInitialize(&argc,&argv,0,help);

        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
	
	// n is the number of division
	PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
	PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
	PetscOptionsGetInt(NULL, NULL, "-its", &max_its, NULL);

	PetscPrintf(comm, "dt=%g\n",dt);
	
        M = n+1;
        double const dx = 1.0/n;
        PetscPrintf(comm, "Matrix size is %d by %d \n", M, M);
        PetscPrintf(comm, "dx=%f\n",dx);
	
	/* convergence requirement */
        if ( dt*k >= 0.5*rou*c*dx*dx ){
                PetscPrintf(comm,"Divergence: dt is not small enough\n");
                exit(0);
        }

	
	// define consts
	double const para1 = dt * k / dx / dx / rou / c;
        double const para2 = dt / rou / c;
        double const pi = PETSC_PI;

	PetscPrintf(comm, "para2=%g\n",para2);
	
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
	// MatView(A,PETSC_VIEWER_STDOUT_WORLD);
	
	
	
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
	// VecView(u,PETSC_VIEWER_STDOUT_WORLD);

	
	/* Duplicate the pattern of u to other vectors */
	VecDuplicate(u,&uold);
        VecDuplicate(u,&f);
	VecDuplicate(u,&diff);
	
	/* Set value for heat supply, f */
	VecGetOwnershipRange(f,&rstart,&rend);
        for (i=rstart; i<rend; i++)
        {
                value_to_set = para2 * sin(l*pi*i*dx);
                VecSetValues(f,1,&i,&value_to_set,INSERT_VALUES);
        }
        VecAssemblyBegin(f);
        VecAssemblyEnd(f);
	// VecView(f,PETSC_VIEWER_STDOUT_WORLD);
		
	VecDuplicate(u,&u_exact);
        VecGetOwnershipRange(u_exact,&rstart,&rend);
        for (i=rstart; i<rend; i++)
        {
                value_to_set = sin(l*pi*i*dx)/pi/pi;
                VecSetValues(u_exact,1,&i,&value_to_set,INSERT_VALUES);
        }
        VecAssemblyBegin(u_exact);
        VecAssemblyEnd(u_exact);	

	/* iteration start */
	for (its=0; its<max_its; its++){
		
			
		VecCopy(u_exact,diff);
                VecAXPY(diff, minus, u);
                VecAbs(diff);
                VecMax(diff, NULL, &err);
		/*
                if ( err < 1.e-8){
                        PetscPrintf(comm, "Converge at %Dth iter, err=%g\n",its,err);
                        break;
                }
		*/

		VecCopy(u,uold);
		MatMultAdd(A, uold, f, u);
		
		/*
		VecNorm(u,NORM_1,&norm_u);
		err = fabs(norm_u - norm_uold);
		
		if ( err < 1.e-8){
                        PetscPrintf(comm, "Converge at %Dth iter, err=%g\n",its+1,err);
                        break;
                }
		norm_uold = norm_u;
		*/
		
	}
	
	// VecView(u,PETSC_VIEWER_STDOUT_WORLD);			
	PetscPrintf(comm, "End at %Dth iter, err=%g\n",its,err);
	
	index = n/2;
	VecGetValues(u, 1, &index, &max_value);
	PetscPrintf(comm, "max_value=%g\n",max_value);
	
	
	
	VecDestroy(&u);
	VecDestroy(&uold);
	VecDestroy(&f);
	VecDestroy(&diff);
	VecDestroy(&u_exact);
	MatDestroy(&A);

        PetscFinalize();
}

