static char help[] = "Assemble the matrix.\n";

#include <stdlib.h>
#include "petscmat.h"
#include <petsc.h>
#include <petscvec.h>
#include <math.h>
#include <petscksp.h>
#include <assert.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
	MPI_Comm        comm;
        PetscMPIInt     rank;
        PetscInt        n=10, max_its=1000;
        PetscInt        i, rstart, rend, its, M, col[3], centre_index, index;
	PetscReal	rou=1.0, c=1.0, k=1.0, dt=0.1, l=1.0, f_to_set, value_to_set, x_coord, err, max_value, value1;
	PetscScalar	value[3], zero=0.0, one=1, minus=-1.0;
	Vec             u, f, uold, u_f, u_exact, diff;
	Mat		A;
	KSP		ksp;
	PetscViewer     h5;

	PetscInitialize(&argc,&argv,0,help);

	PetscViewerCreate(PETSC_COMM_WORLD,&h5);
        comm = MPI_COMM_WORLD;
        MPI_Comm_rank(PETSC_COMM_WORLD,&rank);


        // n is the number of division
        PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL);
	PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
        M = n+1;
	double const dx = 1.0/n;
        PetscPrintf(comm, "Matrix size is %d by %d \n", M, M);
	PetscPrintf(comm, "dx=%f\n",dx);
	PetscPrintf(comm, "dt=%f\n",dt);
	
	// Make sure the value is not 0
	assert(dx);
	assert(dt);
		
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

	if (rstart == 0) rstart = 1;
  
  	if (rend == M)   rend = M-1;

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
	
	
	// Duplicate the pattern of u to other vectors
	VecDuplicate(u,&uold);
        VecDuplicate(u,&u_f);
        VecDuplicate(u,&f);
	VecDuplicate(u,&diff);
	VecDuplicate(u,&u_exact);
	
	// Theoretical solution	
	VecGetOwnershipRange(u_exact,&rstart,&rend);
        for (i=rstart; i<rend; i++)
        {
                value_to_set = sin(l*pi*i*dx)/pi/pi;
                VecSetValues(u_exact,1,&i,&value_to_set,INSERT_VALUES);
        }
        VecAssemblyBegin(u_exact);
        VecAssemblyEnd(u_exact);
		
	// Set value for heat supply, f
	VecGetOwnershipRange(f,&rstart,&rend);
	for (i=rstart; i<rend; i++)
	{
		f_to_set = para2 * sin(l*pi*i*dx);
		VecSetValues(f,1,&i,&f_to_set,INSERT_VALUES);
	}
	VecAssemblyBegin(f);
        VecAssemblyEnd(f);
	// VecView(f,PETSC_VIEWER_STDOUT_WORLD);
	
	VecSet(uold,0.0);
	VecSet(u_f,0.0);

	centre_index = n/2;
	// Create the linear solver
	KSPCreate(PETSC_COMM_WORLD,&ksp);
	KSPSetOperators(ksp,A,A);
        KSPSetTolerances(ksp,1.e-2/M,1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);
        KSPSetFromOptions(ksp);
		
	// Iteration start
	for (its=0; its<max_its; its++){
		
		VecCopy(u_exact,diff);
                VecAXPY(diff, minus, u);
                VecAbs(diff);
                VecMax(diff, NULL, &err);
	
		/* close to theoretical solution */
                if ( err < 1.e-7){
                        PetscPrintf(comm, "Converge at %Dth iter, err=%g\n",its,err);
                        break;
                }
	
		VecCopy(u,uold);	
		VecAXPY(uold, one, f);
		
		// Apply B.C.s	
		VecSetValue(uold, 0, zero, INSERT_VALUES);
		
		VecSetValue(uold, M-1, zero, INSERT_VALUES);

		VecAssemblyBegin(uold);
		VecAssemblyEnd(uold);
		
		KSPSolve(ksp,uold,u);
		
		/* record data in hdf5 */		
		if (its % 100 == 0){
                        PetscViewerHDF5Open(PETSC_COMM_WORLD,"data.h5", FILE_MODE_WRITE, &h5);
                        PetscObjectSetName((PetscObject) u, "implicit-data");
                        VecView(u, h5);
                }
				
	}
	
	PetscPrintf(comm, "End at %Dth iter, err=%g\n",its,err);	

	VecGetValues(u, 1, &centre_index, &max_value);
        PetscPrintf(comm, "centre value=%g\n",max_value);	
	
	/* write result in tecplot */
        FILE *plot = fopen("result.plt","w");
        fprintf(plot, "TITLE=\"1D result\"\n");
        fprintf(plot, "ZONES\"\n");
        fprintf(plot, "VARIABLES = \"X\", \"Y\", \"U\"\n");
        fprintf(plot, "ZONE T=\"2-D Domain\", F=POINT,\n");

        for (int i=0; i<M; i++){
                index = i;
                VecGetValues(u, 1, &index, &value1);
                x_coord = i*dx;
                fprintf(plot, "%g  %g  %g\n", x_coord, 0.0, value1);
        }
		
	// Destroy objects
	PetscViewerDestroy(&h5);
	VecDestroy(&u);
	VecDestroy(&uold);
	VecDestroy(&u_f);
	VecDestroy(&f);
	VecDestroy(&diff);
	VecDestroy(&u_exact);
	MatDestroy(&A);
	KSPDestroy(&ksp);
	
        PetscFinalize();
}
