import numpy as np
import matplotlib.pyplot as plt

# #### 1. Get NumPy Version & Build Info
# Write a Numpy program to get the Numpy version and show the Numpy build configuration.
# print(np.__version__)
# print(np.show_config())


# #### 2. Help for NumPy Add Function
# Write a NumPy program to get help with the add function.
# help(np.add)


# #### 3. Test None are Zero
# Write a NumPy program to test whether none of the elements of a given array are zero. 
# arr = np.array([1, 2, 3, 4, 5, 0])
# arr2 = np.array([1, 2, 3, 4, 5])
# print(arr)
# print(arr2)
# print(np.any(arr == 0))
# print(np.any(arr2 == 0))


# #### 4. Test Any Non-Zero
# Write a NumPy program to test if any of the elements of a given array are non-zero. 
# arr = np.array([1, 0, 0, 0, 0, 0])
# arr2 = np.array([0, 0, 0, 0, 0])
# print(arr)
# print(arr2)
# print(np.any(arr != 0))
# print(np.any(arr2 != 0))


# #### 5. Test Finiteness of Elements
# Write a NumPy program to test a given array element-wise for finiteness (not infinity or not a number).
# arr = np.array([1, np.nan, -np.inf, np.inf, -100])
# print(arr)
# print((~np.isnan(arr)) & (arr != np.inf) & (arr != -np.inf))


# #### 6. Test for Infinity
# Write a NumPy program to test elements-wise for positive or negative infinity. 
# arr = np.array([1, np.nan, -np.inf, np.inf, -100])
# print(arr)
# print((arr == np.inf) | (arr == -np.inf))


# #### 7. Test for NaN
# Write a NumPy program to test element-wise for NaN of a given array.
# arr = np.array([1, np.nan, -np.inf, np.inf, -100])
# print(arr)
# print(np.isnan(arr))


# #### 8. Test for Complex/Real/Scalar
# Write a NumPy program to test element-wise for complex numbers, real numbers in a given array. Also test if a given number is of a scalar type or not. 
# arr = np.array([1, 2 + 3j, 4.2, -5j])
# print(arr)
# print(np.iscomplex(arr))
# print(np.isreal(arr))
# print([np.isscalar(x) for x in arr])


# #### 9. Test Element-Wise Tolerance Equality
# Write a NumPy program to test whether two arrays are element-wise equal within a tolerance.
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([1.00002, 2.00002, 3.001002])
# tolerance = 0.0003
# print(np.abs(arr1 - arr2) <= tolerance)


# #### 10. Element-Wise Comparison (Greater/Less)
# Write a NumPy program to create an element-wise comparison (greater, greater_equal, less and less_equal) of two given arrays.
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([3, 2, 1])

# print(arr1 > arr2)
# print(arr1 >= arr2)
# print(arr1 < arr2)
# print(arr1 <= arr2)

# #### 11. Element-Wise Comparison (Equal/Tolerance)
# Write a NumPy program to create an element-wise comparison (equal, equal within a tolerance) of two given arrays.
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([1.00002, 2.00002, 3.001002])
# tolerance = 0.0003
# print(np.abs(arr1 - arr2) <= tolerance)
# print(arr1 == arr2)

# #### 12. Array Memory Size
# Write a NumPy program to create an array with the values 1, 7, 13, 105 and determine the size of the memory occupied by the array.
# arr = np.array([1, 7, 13, 105])
# print(arr.nbytes)


# #### 13. Create Array of Zeros, Ones, Fives
# Write a NumPy program to create an array of 10 zeros, 10 ones, and 10 fives. 
# print(np.zeros(10))
# print(np.ones(10))
# print(np.full(10, 5))

# #### 14. Create Array of Integers 30 to 70
# Write a NumPy program to create an array of integers from 30 to 70.
# print(np.arange(30, 71))

# #### 15. Create Array of Even Integers 30 to 70
# Write a NumPy program to create an array of all even integers from 30 to 70.
# print(np.arange(30, 71, 2))

# #### 16. Create 3x3 Identity Matrix
# Write a NumPy program to create a 3x3 identity matrix.
# print(np.ones((3, 3)))

# #### 17. Generate Random Number [0,1]
# Write a NumPy program to generate a random number between 0 and 1.
# print(np.random.rand(10))

# #### 18. Generate Random Array from Standard Normal
# Write a NumPy program to generate an array of 15 random numbers from a standard normal distribution.
# print(np.random.standard_normal(15))

# #### 19. Vector 15-55 Without First/Last Values
# Write a NumPy program to create a vector with values ranging from 15 to 55 and print all values except the first and last.
# print(np.arange(15, 56)[1:-1])

# #### 20. Create and Iterate 3x4 Array
# Write a NumPy program to create a 3X4 array and iterate over it.
# mat = np.random.rand(3, 4)
# for item in np.nditer(mat):
#     print(item)

# #### 21. Create Vector of Evenly Spaced Values
# Write a NumPy program to create a vector of length 10 with values evenly distributed between 5 and 50.
# print(np.linspace(5, 50, 10))

# #### 22. Change Sign of Range 9-15
# Write a NumPy program to create a vector with values from 0 to 20 and change the sign of the numbers in the range from 9 to 15.
# arr = np.arange(21)
# print(arr)
# arr[9:16] *= -1
# print(arr)


# #### 23. Vector of Random Integers [0,10]
# Write a NumPy program to create a vector of length 5 filled with arbitrary integers from 0 to 10.
# print(np.random.randint(0, 10, 5))

# #### 24. Multiply Two Vectors`
# Write a NumPy program to multiply the values of two given vectors.`
# arr1 = np.array([2, 4, 6])
# arr2 = np.array([3, 5, 7])
# print(arr1 * arr2)

# #### 25. Create 3x4 Matrix (10 to 21)
# Write a NumPy program to create a 3x4 matrix filled with values from 10 to 21.
# print(np.arange(10, 22).reshape((3, 4)))

# #### 26. Rows & Columns in Matrix
# Write a NumPy program to find the number of rows and columns in a given matrix.
# mat = np.arange(10, 22).reshape((3, 4))
# print(mat.shape)

# #### 27. Create 3x3 Identity Matrix
# Write a NumPy program to create a 3x3 identity matrix, i.e. the diagonal elements are 1, the rest are 0.
# print(np.eye(3))

# #### 28. 10x10 Matrix Border 1, Inside 0
# Write a NumPy program to create a 10x10 matrix, in which the elements on the borders will be equal to 1, and inside 0.
# mat = np.zeros((10, 10))
# mat[[0, -1]] = 1
# mat[:, [0, -1]] = 1
# print(mat)

# #### 29. Diagonal 5x5 Matrix (1-5)
# Write a NumPy program to create a 5x5 zero matrix with elements on the main diagonal equal to 1, 2, 3, 4, 5.
# print(np.diag(np.arange(1, 6)))

# #### 30. Staggered 4x4 Matrix (0,1)
# Write a NumPy program to create a 4x4 matrix in which 0 and 1 are staggered, with zeros on the main diagonal.
# mat = np.fromfunction(lambda x, y: (x + y) % 2, (4, 4))
# print(mat)

# #### 31. Create 3x3x3 Array of Values
# Write a NumPy program to create a 3x3x3 array filled with arbitrary values.
# mat = np.random.randint(0, 27, (3, 3, 3))
# print(mat)

# #### 32. Compute Sums (All, Row, Column)
# Write a NumPy program to compute the sum of all elements, the sum of each column and the sum of each row in a given array.
# mat = np.arange(1, 26).reshape((5, 5))
# print(mat)
# print(mat.sum())
# print(mat.sum(0))
# print(mat.sum(1))


# #### 33. Compute Inner Product of Vectors
# Write a NumPy program to compute the inner product of two given vectors.
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 2, 0])
# print(np.inner(arr1, arr2))

# #### 34. Add Vector to Matrix Rows
# Write a NumPy program to add a vector to each row of a given matrix.\
# mat = np.arange(1, 10).reshape((3, 3))
# arr1 = np.array([1, 2, 3])

# print(mat)
# print(mat + arr1)


# #### 35. Save Array to Binary File
# Write a NumPy program to save a given array to a binary file .
# arr1 = np.array([1, 2, 3])
# arr1.tofile("data.bin")


# #### 36. Save Array and Load from Binary File
# Write a NumPy program to save a given array to a binary file.

# with open('data.bin', 'rb') as f:
#     data = f.read()

# arr = np.frombuffer(data, int)
# print(arr)
# print(type(arr))

# #### 37. Save and Load Array from Text File
# Write a NumPy program to save a given array to a text file and load it.

# arr1 = np.array([1, 2, 3])
# arr1.tofile("data.txt", sep=',')
# arr = np.loadtxt('data.txt', delimiter=',')

# print(arr)
# print(type(arr))


# #### 38. Convert Array to Bytes and Back
# Write a NumPy program to convert a given array into bytes, and load it as an array.
# arr = np.array([1, 2, 3])
# bin_arr = arr.tobytes()
# print(bin_arr)
# print(np.frombuffer(bin_arr, int))


# #### 39. List to Array to List Comparison
# Write a NumPy program to convert a given list into an array, then again convert it into a list. Check initial list and final list are equal or not.
# list = [1, 2, 3]
# arr = np.array(list)
# print(arr)
# back_list = arr.tolist()
# print(back_list)
# print(list == back_list)
# print(id(list) == id(back_list))


# #### 40. Sine Curve Coordinates with Plot
# Write a NumPy program to compute the x and y coordinates for points on a sine curve and plot the points using matplotlib.
# arr = np.linspace(0, 2*np.pi, 1000)
# sins = np.sin(arr)
# # print(sins)
# plt.plot(arr, sins)
# plt.show()

# #### 41. Convert NumPy Dtypes to Python Types
# Write a NumPy program to convert numpy dtypes to native Python types
# num = np.int32(32)
# print(num)
# print(type(num))

# py_num = num.item()
# print(py_num)
# print(type(py_num))

# #### 42. Add Elements Conditionally
# Write a NumPy program to add elements to a matrix. If an element in the matrix is 0, we will not add the element below this element.
# mat = np.arange(25).reshape(5, 5)
# mat[1, 1] = 0
# print(mat)
# mask = (mat == 0).cumsum(axis=0).cumsum(axis=0)
# mask[mask > 2] = 0
# mask[mask == 2] = 1
# mask = mask.astype(bool)
# print(mask)

# mat[mask] = 0
# print(mat.sum(0))

# #### 43. Find Missing Data in Array
# Write a NumPy program to find missing data in a given array.
# arr = np.array([1 ,2 ,np.nan, 4, 5, np.nan])
# print(arr)
# print(np.isnan(arr))

# #### 44. Compare Two Arrays (Element-Wise)
# Write a NumPy program to check whether two arrays are equal (element wise) or not.
# arr1 = np.array([1, 2, 3])
# arr2 = np.array([1, 1, 3])
# print(arr1 == arr2)

# #### 45. Create 1D Array of Digits
# Write a NumPy program to create a one-dimensional array of single, two and three-digit numbers.
# arr1 = np.arange(1, 10)
# print(arr1)

# arr2 = np.arange(10, 100)
# print(arr2)

# arr3 = np.arange(100, 1000)
# print(arr3)

# arr4 = np.concatenate([arr1, arr2, arr3])
# print(arr4)

# #### 46. Create 2D Array of Specified Format
# Write a NumPy program to create a two-dimensional array of a specified format.
# mat = np.random.rand(15).reshape(3, 5)
# print(mat)

# #### 47. Create Array of Random Uniform [0,1]
# Write a NumPy program to create a one-dimensional array of forty pseudo-randomly generated values. Select random numbers from a uniform distribution between 0 and 1.
# arr = np.random.rand(40)
# print(arr)

# #### 48. Random Normal Distribution 2D Array
# Write a NumPy program to create a two-dimensional array with shape (8,5) of random numbers. Select random numbers from a normal distribution (200,7).
# mat = np.random.normal(200, 7, (8, 5))
# print(mat)

# #### 49. Random Sampling with/without Replacement
# Write a NumPy program to generate a uniform, non-uniform random sample from a given 1-D array with and without replacement.
# arr = np.arange(1, 6)

# # Replacement - uniform
# print(np.random.choice(arr, 3))

# # without Replacement - uniform
# print(np.random.choice(arr, 3, replace=False))

# # non-uniform
# weights = np.array([0.5, 0.1, 0.1, 0.1, 0.2])
# print(weights) 
# print(np.random.choice(arr, 3, p=weights))


# #### 50. Swap Rows in 4x4 Random Array
# Write a NumPy program to create a 4x4 array with random values. Create an array from the said array swapping first and last rows.
# mat = np.random.randint(0, 10, (4, 4))
# print(mat)

# mat[[0, -1]] = mat[[-1, 0]]
# print(mat)

# #### 51. Create Zero-Filled Array (5x6)
# Write a NumPy program to create a new array of given shape (5,6) and type, filled with zeros.
# arr = np.zeros((5,6), dtype=float)
# print(arr)

# #### 52. Sort Array by Rows & Columns
# Write a NumPy program to sort a given array by row and column in ascending order.
# mat = np.random.randint(0, 10, (4, 4))
# print(mat)
# mat.sort(axis=0)
# print(mat)
# mat.sort(axis=1)
# print(mat)

# #### 53. Extract Numbers Less/Greater Than Value
# Write a NumPy program to extract all numbers from a given array less and greater than a specified number.
# mat = np.random.randint(0, 10, 10)

# x = 5
# mask = (mat > x) | (mat < x)
# print(mat)
# print(mat[mask])


# #### 54. Replace Numbers Based on Condition
# Write a NumPy program to replace all numbers in a given array equal, less and greater than a given number.
# mat = np.random.randint(0, 10, 10)
# print(mat)
# new_num = -1
# x = 5
# mask1 = mat == 5
# mat[mask1] = new_num
# print(mat)

# mask2 = mat > 5
# mat[mask2] = new_num
# print(mat)

# mask3 = mat < 5
# mat[mask3] = new_num
# print(mat)


# #### 55. Create Array of Same Shape & Type
# Write a NumPy program to create an array of equal shape and data type for a given array.
# mat = np.random.randint(0, 10, 10)
# new_mat = np.ones_like(mat)
# print(new_mat)

# #### 56. Create 3D Array of Shape (3,5,4)
# Write a NumPy program to create a three-dimensional array with the shape (3,5,4) and set it to a variable.
# mat = np.random.randint(0, 10, (3,5,4))
# print(mat)

# #### 57. Swap Columns in 4x4 Array
# Write a NumPy program to create a 4x4 array. Create an array from said array by swapping first and last, second and third columns.
# mat = np.random.randint(0, 10, (4, 4))
# print(mat)
# mat[:, [0, -1]] = mat[:, [-1, 0]]
# mat[:, [1, 2]] = mat[:, [2, 1]]
# print(mat)

# #### 58. Reverse Rows & Columns in Array
# Write a NumPy program to swap rows and columns of a given array in reverse order.
# mat = np.random.randint(0, 10, (3, 4))
# print(mat)
# print(mat[::-1, ::-1].T)

# #### 59. Element-Wise Multiply Two Arrays
# Write a NumPy program to multiply two given arrays of the same size element-by-element.
# arr = np.arange(1, 6)
# arr2 = np.arange(2, 7)

# print(arr)
# print(arr2)
# print(arr * arr2)
