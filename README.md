# HPC_Proj2
This is for a 1D heat conduction problem, in both explicit and implicit form.
explicit_integral.c & implicit_integral.c have all the additional functions, including write in hdf5, write in tecplot form.
The folders under "explicit" performs their own functions for a specific purpose. For example, alpha_dx is to find the value of alpha.
To run the program. First, make main. Then submit the task by script.
Actually, there are some bugs in this implicit coding. The program gives correct results when run in a single processor or a bit of processors. For prosessors more than four, it gives irrational result.
