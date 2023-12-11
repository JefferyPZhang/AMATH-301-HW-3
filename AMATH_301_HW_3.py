import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate

# Problem 1

A = np.genfromtxt('image.csv', delimiter = ',')
U, S, Vt = np.linalg.svd(A, full_matrices = False)

A1 = S
A2 = U.shape
A3 = Vt.shape

S_mat = np.diag(S)

A4 = np.cumsum(S ** 2) / np.sum(S ** 2)
A5 = U[:, 0 : 1] @ S_mat[0 : 1, 0 : 1] @ Vt[0 : 1, :]

def E_Analysis(A4, end):
    k = 0
    while (A4[k] < end):
        k += 1
    return k, A4[k]
end = 0.95
rank, E_Percent = E_Analysis(A4, end)

A6 = rank
A7 = E_Percent
A8 = U[:, 0 : A6] @ S_mat[0 : A6, 0 : A6] @ Vt[0 : A6, :]

end = 0.99
rank, E_Percent = E_Analysis(A4, end)

A9 = rank
A10 = E_Percent
A11 = U[:, 0 : A9] @ S_mat[0 : A9, 0 : A9] @ Vt[0 : A9, :]

# Problem 2

A = np.genfromtxt('CO2_data.csv', delimiter = ',')
x = A[:, 0]
y = A[:, 1]

A12 = x
A13 = y

coeffs = ([300, 30, 0.03])
Sum_Squared_Error = lambda coeffs: np.sum((np.abs(coeffs[0] + coeffs[1] * (np.e ** (coeffs[2] * x)) - y)) ** 2)
coeffs_min = scipy.optimize.fmin(Sum_Squared_Error, coeffs)

A14 = Sum_Squared_Error(coeffs)
A15 = coeffs_min
A16 = Sum_Squared_Error(coeffs_min)

Max_Error = lambda coeffs: np.max(np.abs(coeffs[0] + coeffs[1] * (np.e ** (coeffs[2] * x)) - y))
coeffs_min = scipy.optimize.fmin(Max_Error, coeffs, maxiter = 2000)

A17 = Max_Error(coeffs)
A18 = coeffs_min

coeffs = ([300, 30, 0.03, -5, 4, 0])
Sum_Squared_Error_2 = lambda coeffs: np.sum((np.abs(coeffs[0] + coeffs[1] * (np.e ** (coeffs[2] * x)) + coeffs[3] * np.sin(coeffs[4] * (x - coeffs[5])) - y)) ** 2)

A19 = Sum_Squared_Error_2(coeffs)

coeffs = np.append(A15, [-5, 4, 0])
coeffs_min = scipy.optimize.fmin(Sum_Squared_Error_2, coeffs, maxiter = 2000)

A20 = coeffs_min
A21 = Sum_Squared_Error_2(coeffs_min)

# Problem 3

A = np.genfromtxt('salmon_data.csv', delimiter = ',')
x = A[:, 0]
y = A[:, 1]

A22 = np.polyfit(x, y, 1)
A23 = np.polyfit(x, y, 3)
A24 = np.polyfit(x, y, 5)
A25 = abs(np.polyval(A22, 2022) - 752638) / 752638
A26 = abs(np.polyval(A23, 2022) - 752638) / 752638
A27 = abs(np.polyval(A24, 2022) - 752638) / 752638
A28 = 3