import numpy as np
from autograd import grad, elementwise_grad

# loss_func = lambda p, a: ((p - a) ** 2).mean(0)
# dloss_func = lambda p, a: 2 / len(p) * (p - a)
# a1 = np.random.randn(3, 3)
# a2 = np.random.randn(3, 3)
# print(dloss_func(a1, a2))
# print(elementwise_grad(loss_func)(a1, a2))

l = k = [1, 3,4]

print(k)
l[2] += 2
print(k)