import numpy as np
a = [10, 20, 20, 10, 40, 50]
total = np.sum(np.exp(a))

res = np.exp(a) / total
print(np.sum(res))
