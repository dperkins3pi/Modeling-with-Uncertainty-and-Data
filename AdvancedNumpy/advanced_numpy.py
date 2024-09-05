# advanced_numpy.py
"""Python Essentials: Advanced NumPy.
Daniel Perkins
MATH 437
7/24/24
"""
import numpy as np
from sympy import isprime
from matplotlib import pyplot as plt
import timeit

def prob1(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob1(A)
        array([0, 0, 3])
    """
    A2 = A.copy()
    mask = A2 < 0
    A2[mask] = 0   # Set all negative values to 0
    return A2

def prob2(arr_list):
    """return all arrays in arr_list as one 3-dimensional array
    where the arrays are padded with zeros appropriately."""
    
    # Squeeze all to 2-D arrays
    for i in range(len(arr_list)):   
        arr_list[i] = arr_list[i].squeeze()
        
    # Find the largest number of rows and columns
    m_max, n_max = 0, 0
    for arr in arr_list:
        m = np.shape(arr)[0]
        n = np.shape(arr)[1]
        if m > m_max: m_max = m
        if n > n_max: n_max = n
        
    # Pad arrays with 0s, as needed
    for i in range(len(arr_list)): 
        m =  np.shape(arr_list[i])[0]
        n =  np.shape(arr_list[i])[1]
        if m < m_max:
            zeros = np.zeros((m_max - m, n))
            arr_list[i] = np.vstack((arr_list[i], zeros))
        m = m_max # Now there are more rows
        if n < n_max:
            zeros = np.zeros((m, n_max - n))
            arr_list[i] = np.hstack((arr_list[i], zeros))
    
    # If this problem fails, try changing the dimensions to [1, n, m] 
    # and refer to the Discord comment in labs at  — 07/12/2024 12:36 PM
    return np.dstack(arr_list)

def prob3(func, A):
    """Time how long it takes to run func on the array A in two different ways,
    where func is a universal function.
    First, use array broadcasting to operate on the entire array element-wise.
    Second, use a nested for loop, operating on each element individually.
    Return the ratio showing how many times as fast array broadcasting is than
    using a nested for loop, averaged over 10 trials.
    
    Parameters:
            func -- array broadcast-able numpy function
            A -- nxn array to operate on
    Returns:
            num_times_faster -- float
    """
    # Array broadcasting
    start = timeit.default_timer()
    for i in range(10):
        func(A)
    end = timeit.default_timer()
    time_broadcast = end - start
    
    # For loop
    start = timeit.default_timer()
    for i in range(10):
        for row in A:
            for element in row:
                func(element)
    end = timeit.default_timer()
    time_for_loop = end - start
                
    return time_for_loop / time_broadcast

def prob4(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob4(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
            [ 0.        ,  1.        ,  0.        ],
            [ 0.33333333,  0.33333333,  0.33333333]])
    """
    return A / A.sum(axis=1)  # That was easy
    
# this is provided for problem 5    
def LargestPrime(x,show_factorization=False):
    # account for edge cases.
    if x == 0 or x == 1:
        return np.nan
    
    # create needed variables
    forced_break = False
    prime_factors = [] # place to store factors of number
    factor_test_arr = np.arange(1,11)
    
    while True:
        # a factor is never more than half the number
        if np.min(factor_test_arr) > (x//2)+1:
            forced_break=True
            break
        if isprime(x):  # if the checked number is prime itself, stop
            prime_factors.append(x)
            break
        
        # check if anythin gin the factor_test_arr are factors
        div_arr = x/factor_test_arr
        factor_mask = div_arr-div_arr.astype(int) == 0
        divisors = factor_test_arr[factor_mask]
        if divisors.size > 0: # if divisors exist...
            if divisors[0] == 1 and divisors.size > 1:   # make sure not to select 1
                i = 1 
            elif divisors[0] == 1 and divisors.size == 1:  # if one is the only one don't pick it
                factor_test_arr=factor_test_arr+10
                continue
            else:   # othewise take the smallest divisor
                i = 0
            
            # if divisor was found divide number by it and 
            # repeat the process
            x = int(x/divisors[i])
            prime_factors.append(divisors[i])
            factor_test_arr = np.arange(1,11)
        else:  # if no number was found increase the test_arr 
               # and keep looking for factors
            factor_test_arr=factor_test_arr+10
            continue
    
    if show_factorization: # show entire factorization if desired
        print(prime_factors)
    if forced_break:  # if too many iterations break
        print(f"Something wrong, exceeded iteration threshold for value: {x}")
        return 0
    return max(prime_factors)

def prob5(arr, naive=False):
    """Return an array where every number is replaced be the largest prime
    in its factorization. Implement two methods. Switching between the two
    is determined by a bool.
    
    Example:
        >>> A = np.array([15, 41, 49, 1077])
        >>> prob5(A)
        array([5,41,7,359])
    """
    if naive:
        output = np.zeros_like(arr).astype(np.int32)  # Initalize array with correct type
        for i in range(len(arr)):
            output[i] = LargestPrime(arr[i])  
    else:
        output = np.vectorize(LargestPrime)(arr)  # Vectorize it
    return output


def prob6(x,y,z,A,split=False):
    """Takes three vectors and a matrix and performs 
    (np.outer(x,y)*z.reshape(-1,1))@A on them using einsum.
    If split=True, then the einsum operations should be performed
    one at a time. Otherwise, they should be performed using multiple
    operations at once while using optimize=True. """
    
    if split:
        step_one = np.einsum("i, j -> ij", x, y) # Outer product with x and y
        step_two = np.einsum("ij, i -> ij", step_one, z)   # Array broadcast multiply
        return np.einsum("ij, jk -> ik", step_two, A) # Matrix multiplication
    else:
        return np.einsum("i, j, i, jk -> ik", x, y, z, A, optimize=True)  # All in one step

def naive6(x,y,z,A):
    return np.outer(x,y)*z.reshape(-1,1) @ A  # See end of prob 6

def prob7():
    """Times and generates a plot that compares the difference in
    speeds between Einsum operations and NumPy.
    """
    # Initialize arrays for timing
    ein_split = []
    ein_opt = []
    nump = []
    ns = np.arange(3, 501)
    for n in ns:  # For n=3 to n=500
        x = np.random.rand(n)
        y = np.random.rand(n)
        z = np.random.rand(n)
        A = np.random.rand(n,n)
        # Time the einsum split
        start = timeit.default_timer()
        prob6(x,y,z,A,split=True)
        ein_split.append(timeit.default_timer() - start)
        # Time the einsum optimized
        start = timeit.default_timer()
        prob6(x,y,z,A,split=False)
        ein_opt.append(timeit.default_timer() - start)
        # Time the numpy function
        start = timeit.default_timer()
        naive6(x,y,z,A)
        nump.append(timeit.default_timer() - start)
        
    # Plot it 
    plt.title("Einsum vs. NumPy")
    plt.xlabel("Input Size")
    plt.ylabel("Output size")
    plt.plot(ns, ein_split, label="Einsum Split")
    plt.plot(ns, ein_opt, label="Einsum Optimized")
    plt.plot(ns, nump, label="NumPy")
    plt.legend()
    plt.show()
    


if __name__=="__main__":
    # Prob 1
    # A = np.array([-3,-1,3])
    # print(prob1(A))
    
    # Prob 2
    # A = np.array([[[1], [2]], [[3], [4]]])
    # B = np.array([[[5], [6], [7]], [[8], [9], [10]]])
    # C = np.array([[[11], [12]], [[13], [14]], [[15], [16]]])
    # arr_list = [A, B, C]
    # print(arr_list)
    # print(prob2(arr_list))
    
    # Prob 3
    # func = np.log
    # A = np.random.rand(500, 500)
    # print(prob3(func, A))
    
    # Prob 4
    # A = np.array([[1,1,0],[0,1,0],[1,1,1]])
    # print(prob4(A))
    
    # Prob 5
    # A = np.array([15, 41, 49, 1077])
    # print(prob5(A, naive=True))
    # print(prob5(A))
    
    # Prob 6
    # x = np.array([1, 2]).astype(np.float64)
    # y = np.array([3, 4]).astype(np.float64)
    # z = np.array([-1, 1]).astype(np.float64)
    # A = np.array([[1, 2],
    #             [3, 4]]).astype(np.float64)
    # print("optimized", prob6(x,y,z,A))
    # print("naive", naive6(x,y,z,A))
    
    # Prob 7
    # prob7()
    
    pass