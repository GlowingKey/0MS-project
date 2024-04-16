import numpy as np
import plotly.graph_objects as go
import sys
import plotly.io as pio
import time


def read_matrix() -> np.ndarray[np.ndarray]:
    """
    Function reads a matrix called A from the input to solve a linear equation in the 'Ax = b' form
    :return: np.ndarray[np.ndarray] - matrix A from the input
    """
    inp = list(map(lambda x: x.strip(' \n'), sys.stdin.readlines()))
    inp = list(map(lambda x: x.split(), inp))

    return np.array([np.array(row, dtype=int) for row in inp])


def read_vector() -> np.ndarray:
    """
    Function reads a vector called b from the input to solve a linear equation in the 'Ax = b' form
    :return: np.ndarray - vector b from the input
    """
    return np.array(list(map(int, input("Enter vector b in format 'x y z': ").split())))


def check_dimensions(a: np.ndarray[np.ndarray], b: np.ndarray) -> bool:
    if len(a) != len(b):
        print("vector b should be in the same dimension as the matrix A")
        return False
    return True


def find_projection(a: np.ndarray[np.ndarray], b: np.ndarray) -> np.ndarray:
    """
    Function finds the projection vector of a vector b onto the column space of a matrix A.
    If vector b is in the column space of matrix A, then we can solve the equation Ax = b.
    If vector b is not in the column space of A, we need to project a vector b onto this column space to find the 'closest' solution.
    :param a: matrix A
    :param b: vector b
    :return: np.ndarray - a projection vector of a vector b
    """

    # To find a projection vector we need to find a projection matrix:
    #  p = A * x = P * b => P = A * (A^T * A)^-1 * A^T

    # Calculate A^T * A
    ATA = np.dot(a.T, a)
    # Calculate (A^T A)^(-1)
    ATA_inv = np.linalg.inv(ATA)

    matrix_p = np.dot(np.dot(a, ATA_inv), a.T)
    return np.round(np.dot(matrix_p, b), decimals=1)


def solve_lin_equation(a: np.ndarray[np.ndarray], p: np.ndarray) -> np.ndarray:
    """
    Function uses the Least squares method to find a solution for our new vector p.
    :param a:  matrix A
    :param p: vector p
    :return: np.ndarray - solution for the system A*new_x = p
    """

    # A^T * A * new_x = A^T * p => new_x = (A^T * A)^-1 * (A^T * p)
    # Calculate A^T * A
    ATA = np.dot(a.T, a)
    # Calculate (A^T A)^(-1)
    ATA_inv = np.linalg.inv(ATA)

    return np.dot(ATA_inv, np.dot(a.T, p))


def plot_dimension(a: np.ndarray[np.ndarray], b: np.ndarray, p: np.ndarray) -> None:
    """
    Plots a 3d plane given by combinations of  vectors of matrix A, vector b and its projection
    to graphically demonstrate the problem and its solution.
    :param a: matrix A
    :param b: vector b
    :param p: projection of the vector b
    """

    # plane equation is a*x + b*y + c*z = 0
    # get cross product of the vector of matrix A
    cross_prod = np.cross(a[:, 0], a[:, 1])

    # get points for the plane
    x = np.linspace(-a[:, 0].max(), a[:, 0].max(), 500)
    y = np.linspace(-a[:, 1].max(), a[:, 1].max(), 500)

    X, Y = np.meshgrid(x, y)
    # get z from the equation of a plane
    Z = -(cross_prod[0] / cross_prod[2]) * X - (cross_prod[1] / cross_prod[2]) * Y

    # plot a surface in 3D
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

    fig.update_traces(
        colorscale='Viridis',  # color
        showscale=False,  # Remove the color bar
        opacity=0.6  # Set the transparency level
    )

    # plot vector 'b'
    fig.add_trace(go.Scatter3d(
        x=[0, b[0]],
        y=[0, b[1]],
        z=[0, b[2]],
        mode='lines+markers',  # Draw line segment with markers
        marker=dict(size=8),  # Marker size
        line=dict(color='black', width=3),  # Line color and width
        name='Vector b'
    ))

    # plot vector 'p' (projection)
    fig.add_trace(go.Scatter3d(
        x=[0, p[0]],
        y=[0, p[1]],
        z=[0, p[2]],
        mode='lines+markers',  # Draw line segment with markers
        marker=dict(size=8),  # Marker size
        line=dict(color='red', width=3),  # Line color and width
        name='Projection of vector b'
    ))

    # pio.show(fig,validate=False)
    fig.show()


def main():
    b = read_vector()
    print()
    print("Enter 3x2 matrix A:")
    A = read_matrix()
    if check_dimensions(A, b):
        p = find_projection(A, b)
        print(f'projection vector = {p}')
        new_x = solve_lin_equation(A, p)
        print(f'solution of the system of linear equations = {new_x}')
        plot_dimension(A, b, p)
        time.sleep(100000)


if __name__ == "__main__":
    main()
