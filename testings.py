import mxnet as mx
import gluoncv as gcv

image_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/mt_baker.jpg'
image_filepath = 'mt_baker.jpg'
gcv.utils.download(url=image_url, path=image_filepath)

# def generate_primes(n):
#
#     if n < 2:
#         return []
#     size = (n-3)
#     primes = [2]
#     is_prime = [True] * size
#     for i in range(size):
#         if is_prime[i]:
#             p = i * 2 + 3
#             primes.append(p)
#             for j in range(2 * i**2 + 6*i + 3, size,p):
#                 is_prime[j] = False
#     return primes
#
# print(generate_primes(100))
#
# a = mxn.nd.array([1,2,3,4,5])
#
# !ls mt_baker.jpg