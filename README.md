# 0MS project

This project solves system of linear equations in the form Ax = b, where b is a 3x1 vector and it is NOT a linear combination of the vectors of matrix(3x2) A, i.e., this vector b is not in the column space of A.  
So, such a system doesn't have a solution. However, we can find 'the nearest' vector to vector b, that is in the column space of A.
For this purpose we will use a projection of vector b onto a subspace created by vectors of matrix A. 
So, now we have a system Ax = p, where p is a projection of vector b onto matrix A. 

Now we can use the Least Squares method to find a solution for our new vector p. To do so, matrix A should be a non-singular matrix.

This program is able to find solution for any 3x2 non-singular matrix A and 3x1 vector b. It also provides a nice graph, representing the principle of the projection, and the reason we can't solve a system for a vector b, that is not in the column space of A.

### RUN
1. Create a virtual environment: <code>python -m venv env </code>
2. Activate the created environment(for Mac/Linux): <code>source env/bin/activate</code>
3. Install required libraries : <code>pip install -r requirements.txt</code>

### Input arguments:
First argument is a 3x1 vector b in the form: x y z; <br>
Second argument is a 3x2 matrix A (one row on each line); <br>
To stop entering elements of a matrix just push Ctrl + D.

Examples of matrices and vectors for tests are available in the file "examples.txt".