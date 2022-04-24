import math
import numpy as np

# print(sum(np.array([1, 1, 1]) == np.array([1, 0, 1])))

# for i in range(len(y)):
#     if y[i] * np.inner(w, X[i]) <= 0:
#         temp_w = temp_w + y[i] * X[i]
#         c += 1
#         break
# if np.array_equal(temp_w, w):
#     self.coefs_ = temp_w
#     self.fitted_ = True
#     break
# w = temp_w
# self.coefs_ = temp_w
# self.fitted_ = True

x = np.array([1, 3, -4, 3])
y = np.zeros(shape=len(x))
# print(np.diag(x))
# print(np.where(np.any(x <= y)))
# z = np.where(np.any(x <= y))
# print(x[z])
y = np.array([[1, 2, 3], [4, 5, 6]])
#print(type(np.amax(y, axis=1)))

# A = self._cov_inv @ self.mu_.T
# At_Xt = A.T @ X.T
# c = 0
# for mu_i in self.mu_:
        #     At_Xt[c] += mu_i @ self._cov_inv @ mu_i.T
        #     c += 1
# return np.amax(At_Xt, axis=0)
y  =  np.array([[2,3,2],[3,4,5]])
#print(np.var(y[(0,1),0]))
print(np.amax(y, axis=0))