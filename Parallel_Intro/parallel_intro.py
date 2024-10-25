# iPyParallel - Intro to Parallel Programming

import time
import numpy as np
from ipyparallel import Client
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as sparse on
    all engines. Return the DirectView.
    """
    client = Client()  # Create a Client object
    dview = client[:]  # Create a Direct view with all engines
    dview.block = True
    dview.execute("import scipy.sparse as sparse")
    client.close()

# Problem 2
def prob2(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    def f(n):  # Function that samples and gets the mean, min, max
        x = np.random.normal(0, 1, n)
        return np.mean(x), np.min(x), np.max(x)
    
    client = Client()  # Create a Client object
    dview = client[:]  # Create a Direct view with all engines
    dview.block = True
    
    dview.execute("import numpy as np")  # Apply function to each process
    output = dview.apply(f, n)
    
    # Correct output format
    means = [result[0] for result in output]
    mins = [result[1] for result in output]
    maxs = [result[2] for result in output]
    
    client.close() # Close client
    
    return means, mins, maxs

# Problem 3
def prob3():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 2 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    ns = [1000000, 5000000, 10000000, 15000000]
    times_par = []
    for n in ns:  
        # Time prob2 for each n
        start = time.time()
        means, mins, maxs = prob2(n)
        end = time.time()
        times_par.append(end - start)
        
    N = len(means)  # Determine the number of engines in the machine
    
    times_ser = []
    for n in ns:
        # Time serial version
        start = time.time()
        for i in range(N):
            x = np.random.normal(0, 1, n)
            np.mean(x)
            np.min(x)
            np.max(x)
        end = time.time()
        times_ser.append(end - start)
    
    print("grjn")
    # Plot the stuff
    plt.plot(ns, times_par, label="Parallel")
    plt.plot(ns, times_par, label="Serial")
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("time (s)")
    plt.show()

# Problem 4
def parallel_trapezoid_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    xs = np.linspace(a, b, n)
    client = Client()  # Create a Client object
    dview = client[:]  # Create a Direct view with all engines
    dview.block = True
    N = len(client)    # Determine the number of agents
        
    # Split up the array so that the endpoints are shared
    chunks = np.array_split(xs, N)  # Split array into N chunks
    new_chunks = []
    for i in range(len(chunks)):  # Add endpoints
        chunk = chunks[i]
        if i > 0: chunk = np.insert(chunk, 0, chunks[i-1][-1])
        new_chunks.append(chunk)
        
    def trapezoidal_rule(xs):
        sum = 0
        h = xs[1] - xs[0]
        m = len(xs)
        for i in range(m - 1):
            sum += f(xs[i]) + f(xs[i+1])  # Add things in the sum
        return (h / 2) * sum
    
    dview.execute("import numpy as np")
    dview.scatter("xs_small", new_chunks)  # Scatter the data to each engine
    dview.push({"trapezoidal_rule": trapezoidal_rule})   # Send the function to each
    dview.execute("sum = trapezoidal_rule(xs_small[0])")
    outputs = dview["sum"]    
    client.close() # Close client
    
    return np.sum(outputs)


if __name__=="__main__":
    # Problem 1
    # prob1()
    plt.plot()
    plt.show()
    
    # Problem 2
    # means, mins, maxs = prob2()
    # print(means)
    # print(mins)
    # print(maxs)
    
    # Prob 3
    # prob3() # PLOTTING NOT WORKING ANYMORE!!!!!!!!!!!!
    
    # Prob 4
    # def f(x): return 3*x**2 - 2*np.sin(x) + 5.3
    # a = -4
    # b = 7
    # sum = 0
    # n = 100
    # xs = np.linspace(a, b, n)
    # h = xs[1] - xs[0]
    # for i in range(n - 1):
    #     sum += f(xs[i]) + f(xs[i+1])
    # print("Serial method:", (h / 2) * sum)
    # print("Parallel method:", parallel_trapezoid_rule(f, a, b, n=n))
    
    pass