import numpy as np

# Full blind search method in Python
def fsearch(search, fn, type="min", **kwargs):
    x = np.apply_along_axis(fn, 1, search, **kwargs)
    if type == "min":
        ib = np.argmin(x)
    else:
        ib = np.argmax(x)
    return {"index": ib, "sol": search[ib], "eval": x[ib]}

# Depth-first full search method in Python
def dfsearch(l=1, b=1, domain=None, fn=None, type="min", D=None, par=None, bcur=None, **kwargs):
    if par is None:
        par = [None] * D
    if domain is None:
        domain = [[] for _ in range(D)]
    if bcur is None:
        bcur = {'sol': None, 'eval': float('inf') if type == 'min' else float('-inf')}

    if (l - 1) == D:
        f = fn(par, **kwargs)
        fb = bcur['eval']
        if (type == 'min' and f < fb) or (type == 'max' and f > fb):
            return {'sol': par, 'eval': f}
        return bcur
    else:
        for j in domain[l - 1]:
            par[l - 1] = j
            bcur = dfsearch(l + 1, b, domain, fn, type, D, par, bcur, **kwargs)
        return bcur
    
    
import numpy as np

# Helper function to convert an integer to a binary array
def binint(x, D):
    return np.array(list(np.binary_repr(x, width=D))).astype(int)

# Convert a binary array to an integer
def intbin(x):
    return int("".join(map(str, x)), 2)

# Evaluation function: sum of binary values
def sumbin(x):
    return np.sum(x)

# Evaluation function: max sin of binary values
def maxsin(x, Dim):
    return np.sin(np.pi * intbin(x) / (2 ** Dim))

D = 8  # Number of dimensions
x = np.arange(2 ** D)  # Integer search space

# Setting up the full search space
search = np.array([binint(xi, D) for xi in x])

# Setting the domain values (D binary variables)
domain = [[0, 1] for _ in range(D)]

# Full blind search for sum of bits
S1 = fsearch(search, sumbin, "max")
print(f"fsearch best s: {S1['sol']} f: {S1['eval']}")

# Depth-first search for sum of bits
S2 = dfsearch(domain=domain, fn=sumbin, type="max", D=D)
print(f"dfsearch best s: {S2['sol']} f: {S2['eval']}")

# Full blind search for max sin
Dim = len(search[0])  # Dimension for maxsin function
S3 = fsearch(search, maxsin, "max", Dim=Dim)
print(f"fsearch best s: {S3['sol']} f: {S3['eval']}")

# Depth-first search for max sin
S4 = dfsearch(domain=domain, fn=maxsin, type="max", D=D, Dim=Dim)
print(f"dfsearch best s: {S4['sol']} f: {S4['eval']}")

# Step 1: Convert integer 7 to its binary representation and select the first 4 bits.
"{0:b}".format(7)
"{0:b}".format(7)[0:3]
intbin("{0:b}".format(7)[1:4])

selected_bits = binary_x[-4:]
print("Original bits:", selected_bits)

x = x[::-1]
"{0:b}".format(7)

print("Reversed bits:", x.bin)

x_values = [bit for bit in x]

print("Converted values:", x_values)
