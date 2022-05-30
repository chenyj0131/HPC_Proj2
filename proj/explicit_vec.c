static char help[] = "Explicit algorithm using vector\n";

#include <stdlib.h>
#include "petscmat.h"
#include <petsc.h>
#include <petscvec.h>
#include <math.h>

int main(int argc,char **argv)
{
	MPI_Comm        comm;
        PetscMPIInt     rank;
        PetscInt        n=10, max_its=4000;
	PetscInt	i, rstart, rend, its, M, index_uold[3], index_f[1];
	PetscReal	rou=1.0, c=1.0, k=1.0, dt=0.001, l=1.0, norm_uold=0.0, norm_u=1.0, value;
	PetscScalar	zero=0.0, f_to_add, y[3];

	Vec		u, uold, f;
	double		err;


	// The first column is the type of bcs, 1 for set value, 2 for heat flux
	// The firt row represents point at left end
	
	// Initialize and get the rank
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
	
	// define the consts
	double const para1 = dt * k / dx / dx / rou / c;
        double const para2 = dt / rou / c;
        double const pi = PETSC_PI;
	
	// Create vector f. Local dimension is default.
	VecCreate(PETSC_COMM_WORLD,&f);
        VecSetSizes(f,PETSC_DECIDE,M);
        VecSetFromOptions(f);
	// set value for f
	VecGetOwnershipRange(f,&rstart,&rend);
	for (i=rstart; i<rend; i++){
		value = para2 * sin(l*pi*i*dx);
		VecSetValue(f,i,value,INSERT_VALUES);
	}
	
	VecAssemblyBegin(f);
        VecAssemblyEnd(f);
	PetscPrintf(comm, "view f:\n");
        VecView(f,PETSC_VIEWER_STDOUT_WORLD);
	
	// Duplicate the pattern of u to other vectors
	VecDuplicate(f,&uold);
        VecDuplicate(f,&u);
        // set value for u, and used for initial value.
	VecGetOwnershipRange(u,&rstart,&rend);
        for (i=rstart; i<rend; i++){
                value = exp(i*dx);
                VecSetValue(u,i,value,INSERT_VALUES);
        }

        VecAssemblyBegin(u);
        VecAssemblyEnd(u);
	PetscPrintf(comm, "view u0:\n");
        VecView(u,PETSC_VIEWER_STDOUT_WORLD);
	
	
	VecGetOwnershipRange(u,&rstart,&rend);
        PetscPrintf(PETSC_COMM_SELF, "rank [%d]: istart = %d iend = %d \n", rank, rstart, rend );
        if (rstart==0)  rstart = 1;
        if (rend == M)  rend   = M-1;
        PetscPrintf(PETSC_COMM_SELF, "rank [%d]: istart = %d iend = %d \n", rank, rstart, rend );

	// Iteration start
	for (its=0; its<max_its; its++){
		VecCopy(u,uold);
		
		// calculate each number in u_new
		for (i=rstart; i<rend; i++){
			//PetscPrintf(PETSC_COMM_SELF, "\ni:%d \n",i );
			
			// get three values from uold
			index_uold[0]=i-1; index_uold[1]=i; index_uold[2]=i+1;
			VecGetValues(uold, 3, index_uold, y);
			
			// get one value from f
			index_f[0] = i;
			VecGetValues(f,    1, index_f, &f_to_add);
			
			value = para1*y[0] + (1.0-2.0*para1)*y[1] + para1*y[2] + f_to_add;
			VecSetValue(u,i,value,INSERT_VALUES);
		}
		
		VecSetValue(u,0,zero,INSERT_VALUES);
		VecSetValue(u,M-1,zero,INSERT_VALUES);
		VecAssemblyBegin(u);
	        VecAssemblyEnd(u);
			
		VecNorm(u,NORM_1,&norm_u);
                err = fabs(norm_u - norm_uold);
                if ( err < 1.e-8){
                        PetscPrintf(comm, "End at %Dth iter, err=%g\n",its,err);
                        break;
                }
                norm_uold = norm_u;
		
	}

	VecView(u,PETSC_VIEWER_STDOUT_WORLD);
	
	
	VecDestroy(&u);
        VecDestroy(&uold);
        VecDestroy(&f);

        PetscFinalize();
}
	









