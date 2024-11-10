import numpy as np

x = np.ones((8, 8))
print('Before:')
print(x)

# Your code goes here
x[1:7,1:7] = 0
print('After:') 
print(x)