import numpy as np

import main

# Test setting up the galois fields and multiplication (genai test)
def test_galois_field(w):
    gflog, gfilog, prim_poly = main.setup_tables(w)

    # Test 1: Ensure gflog[gfilog[a]] == a for all a in GF(2^w)
    for a in range(1, (1 << w) - 1):  # Exclude 0 as it has no logarithm
        assert gflog[gfilog[a]] == a, f"Test failed for a = {a}"

    # Test 2: Ensure gfilog[gflog[a]] == a for all a in GF(2^w)
    for a in range(1, (1 << w) - 1):  # Exclude 0 as it has no logarithm
        assert gfilog[gflog[a]] == a, f"Test failed for a = {a}"

    # Test 3: Verify multiplication property a * b = gfilog[gflog[a] + gflog[b]]
    for a in range(1, (1 << w) - 1):
        for b in range(1, (1 << w) - 1):
            expected = main.gf_multiply(a, b, w, prim_poly)
            result = gfilog[(gflog[a] + gflog[b]) % ((1 << w) - 1)]
            assert result == expected, f"Multiplication failed for a = {a}, b = {b}. Expected {expected}, got {result}"

    print(f"All tests passed for GF(2^{w})!")


test_galois_field(8)
test_galois_field(4)

def test_vandermonde():
    w = 4
    gflog, gfilog, prim_poly = main.setup_tables(w)
    m = 3
    n = 3
    result = main.setup_vandermonde(m, n, w, prim_poly)
    expected = [[1, 1, 1], [1, 2, 3], [1, 4, 5]]  # from paper example in section 'An example'
    assert result == expected, f"Vandermonde table failed for a {m}x{n} table and w = {w}." \
                               f" Expected {expected}, got {result}."
    print(f"Vandermode table successfully constructed")


test_vandermonde()

def test_steps3thru5():
    w = 4
    m = 3
    n = 3
    gflog, gfilog, prim_poly = main.setup_tables(w)
    F = main.setup_vandermonde(m, n, w, prim_poly)
    D = [3, 13, 9]
    C = main.calculate_checksums(F, D, w, prim_poly)
    expected_C = [7, 2, 9]  # from paper example in section 'An example'
    assert C == expected_C, f'Checksum portion failed. Expected {expected_C}, got {C}.'
    print(f"Checksum generation passed")
    A, E = main.generate_ae(F, D, C)
    A_expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 2, 3], [1, 4, 5]]
    E_expected = [[3], [13], [9], [7], [2], [9]]
    assert np.array_equal(np.array(A_expected), A), f'A creation differed: Expected {A_expected}, got {A}.'
    assert np.array_equal(np.array(E_expected), E), f'E creation differed: Expected {A_expected}, got {A}.'
    print(f"AE Generation (creation step) works")
    A_prime, E_prime = main.generate_ae_prime(F, D, C, [1, 2, 5])
    A_expected = [[1, 0, 0], [1, 1, 1], [1, 2, 3]]
    E_expected = [[3], [7], [2]]
    assert np.array_equal(np.array(A_expected), A_prime), f'A creation differed: Expected {A_expected}, got {A_prime}.'
    assert np.array_equal(np.array(E_expected), E_prime), f'E creation differed: Expected {A_expected}, got {E_prime}.'
    print(f"AE Prime (deletion step) Generation works")


test_steps3thru5()


