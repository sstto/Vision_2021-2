import numpy as np


# h1 = 318
# w1 = 636
# y_flat = np.repeat(range(h1), w1)
# x_flat = np.tile(range(w1), h1)
# yx_flat = np.concatenate((y_flat.reshape(-1, 1), x_flat.reshape(-1, 1)), axis=1)
#
# print(yx_flat)

x, y = 1, 1
jacobian = np.array([[x, 0, y, 0, 1, 0], [0, x, 0, y, 0, 1]])


a = np.array([3, 4])

print(a.dot(jacobian))