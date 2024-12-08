import numpy as np

# Step 1 : Choose Parameters
# given n data devices and m checksum devices is tolerant up to m failures as follows
w = 8  # 2^w > n + m, easiest is w=8 or w=16. For w=8 n + m can be up to 255 and for w=16 n + m can be up to 65535.
m = 3
n = 4

# step 2 - set up gflog and gfilog
def setup_tables(w):
    # Define the primitive polynomials for different field sizes
    prim_poly_4 = 0o23  # 4-bit field (octal representation)
    prim_poly_8 = 0o435  # 8-bit field (octal representation)
    prim_poly_16 = 0o210013  # 16-bit field (octal representation)

    # Select the appropriate primitive polynomial based on the field size (w)
    if w == 4:
        prim_poly = prim_poly_4
    elif w == 8:
        prim_poly = prim_poly_8
    elif w == 16:
        prim_poly = prim_poly_16
    else:
        raise ValueError("Unsupported field size")

    # Initialize the gflog and gfilog arrays
    x_to_w = 1 << w  # Field size: 2^w
    gflog = [0] * x_to_w
    gfilog = [0] * x_to_w

    b = 1
    for log in range(x_to_w - 1):
        gflog[b] = log
        gfilog[log] = b
        b = b << 1
        if b & x_to_w:
            b ^= prim_poly  # Apply the primitive polynomial

    return gflog, gfilog, prim_poly

def gf_multiply(a, b, w, prim_poly):
    result = 0
    for i in range(w):
        if (b & (1 << i)) != 0:
            result ^= a << i

    # Reduce the result modulo the primitive polynomial
    for i in range(2 * w - 1, w - 1, -1):
        if (result & (1 << i)) != 0:
            result ^= prim_poly << (i - w)

    return result & ((1 << w) - 1)  # Ensure result fits in w bits

def gf_power(base, exp, w, prim_poly):
    """Compute base^exp in GF(2^w), reduced by the primitive polynomial."""
    result = 1
    while exp > 0:
        if exp % 2 == 1:
            result = gf_multiply(result, base, w, prim_poly)  # Multiply by base if the exponent is odd
        base = gf_multiply(base, base, w, prim_poly)  # Square the base
        exp //= 2  # Halve the exponent
    return result

# step 3
def setup_vandermonde(m, n, w, prim_poly):
    """Set up the Vandermonde matrix F for GF(2^w)."""
    # Initialize an empty Vandermonde matrix (m x n)
    F = [[0] * n for _ in range(m)]

    # Fill the Vandermonde matrix with j^(i-1) values mod (2^w)
    for i in range(m):
        for j in range(n):
            # The (i,j)-th element is j^(i-1) mod (2^w)
            F[i][j] = gf_power(j + 1, i, w, prim_poly)
    return F


def calculate_checksums(F, data_words, w, prim_poly):
    '''
    We essentially are calculating F * d = c
    '''
    m = len(F)  # Number of checksum devices (rows of F) m
    n = len(F[0])  # Number of data devices (columns of F) n
    checksums = [0] * m  # Initialize checksum words
    for i in range(m):  # For each checksum device (row in F)
        for j in range(n):  # For each data device
            # Perform multiplication and addition in GF(2^w)
            checksums[i] ^= gf_multiply(F[i][j], data_words[j], w, prim_poly)
    return checksums


def generate_ae(F, D, C):
    """
    :param F: m x n Vandermonde matrix
    :param D: data 1d array
    :param C: checksum 1d array
    :param failed_devices: 1d array of (0-based) indices of rows to be deleted
    :return: A' and E' that can be used to solve for data given that less than m rows were deleted
    """
    # Construct the identity matrix I with shape (n, n) assuming D is a vector of size n
    D = np.array(D)
    C = np.array(C)
    I = np.eye(len(D))
    # per paper: A = [I,
    #                 F]
    #            E = [D,
    #                 C]
    A = np.vstack([I, F])  # Stack identity matrix I and Vandermonde matrix F vertically
    E = np.vstack([D.reshape(-1, 1), C.reshape(-1, 1)])  # Concatenate D and C vertically
    return A, E

def generate_ae_prime(F, D, C, failed_devices):
    """
    :param F: m x n Vandermonde matrix
    :param D: data 1d array
    :param C: checksum 1d array
    :param failed_devices: 1d array of (0-based) indices of rows to be deleted
    :return: A' and E' that can be used to solve for data given that less than m rows were deleted
    """
    A, E = generate_ae(F, D, C)
    A_prime = np.delete(A, failed_devices, axis=0)
    E_prime = np.delete(E, failed_devices, axis=0)
    # Delete extra rows to ensure we have exactly n rows
    if A_prime.shape[0] > len(D):
        A_prime = A_prime[:len(D), :]
        E_prime = E_prime[:len(D), :]
    return A_prime, E_prime

def generate_random_data(n, w):
    """
    Generate random data for the n data devices, values between 1 and 2^w - 1.
    :param n: number of data points
    :param w: range
    :return: length d array with random values between 1 and 2^w - 1
    """
    return [np.random.randint(1, (1 << w) - 1) for _ in range(n)]


def solve_in_gf(A, b, w, prim_poly):
    """
    Solve the linear system A * x = b in GF(2^w) using Gaussian elimination.
    :param A: Coefficient matrix (n x n)
    :param b: Right-hand side vector (n)
    :param w: Field size parameter for GF(2^w)
    :param prim_poly: Primitive polynomial for GF(2^w)
    :return: Solution vector x
    """
    n = len(b)
    A = A.astype(int).copy()  # Make a copy of A to modify
    b = b.astype(int).copy()  # Make a copy of b to modify

    # Forward elimination
    for i in range(n):
        # Find the pivot element
        if A[i][i] == 0:
            for j in range(i + 1, n):
                if A[j][i] != 0:
                    A[[i, j]] = A[[j, i]]  # Swap rows
                    b[i], b[j] = b[j], b[i]
                    break

        # Normalize the pivot row
        pivot_inv = gf_power(A[i][i], (1 << w) - 2, w, prim_poly)  # Inverse in GF(2^w)
        for j in range(i, n):
            A[i][j] = gf_multiply(A[i][j], pivot_inv, w, prim_poly)
        b[i] = gf_multiply(b[i], pivot_inv, w, prim_poly)

        # Eliminate the current column in the rows below
        for j in range(i + 1, n):
            factor = A[j][i]
            for k in range(i, n):
                A[j][k] ^= gf_multiply(factor, A[i][k], w, prim_poly)
            b[j] ^= gf_multiply(factor, b[i], w, prim_poly)

    # Back substitution
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] ^= gf_multiply(A[i][j], x[j], w, prim_poly)
    return x

def solve_for_data(F, D, C, failed_devices, w, prim_poly):
    """
    Solve for the missing data using the reduced A' matrix and E' vector.
    :param F: Vandermonde matrix (m x n)
    :param D: Original data vector
    :param C: Checksum vector
    :param failed_devices: Indices of rows that correspond to failed devices
    :param w: GF(2^w) field size
    :param prim_poly: Primitive polynomial for GF(2^w)
    """
    if len(failed_devices) > len(C):
        raise Exception("Too many devices failed")
    # Generate reduced A' and E', which gets rid of failed_devices information
    A_prime, E_prime = generate_ae_prime(F, D, C, failed_devices)
    # Solve for the missing data using Gaussian elimination in GF(2^w)
    recovered_data = solve_in_gf(A_prime, E_prime.flatten(), w, prim_poly)
    recovered_data = np.array(recovered_data)
    return recovered_data


if __name__ == '__main__':
    gflog, gfilog, prim_poly = setup_tables(w)
    F = setup_vandermonde(m, n, w, prim_poly)
    D = generate_random_data(n, w)
    print(f'Generated data is: {np.array(D)}')
    C = calculate_checksums(F, D, w, prim_poly)
    print(f'Generated checksum is {np.array(C)}')
    ### Going to try and restore the data given that we only have d_1, d_2
    recovered_data = solve_for_data(F, D, C, [1, 2, 3], w, prim_poly)
    print(f"Recovered data is {recovered_data}")


