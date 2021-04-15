# def count_bits(x):
#     num_bits = 0
#     while x:
#         num_bits += x & 1
#         x >>= 1
#     return num_bits
#
# print(count_bits(12))


# import numpy
# print(numpy.version.version)

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("bacall.oldd.jpg")

print(type(img))
print(np.shape(img))

plt.show()
plt.imshow(img, cmap='gray')