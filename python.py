import numpy as np
import test_module

print(dir(test_module))

a = np.zeros((1, 1), dtype=np.int8)

test_module.greet(a)

a = None

b = test_module.greet(np.zeros((0,)))

print(b)
