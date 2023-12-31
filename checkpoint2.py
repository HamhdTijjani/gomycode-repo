import numpy as np
import math

#Q1: A program that will find number which are divisible by 7 but not a multiple of 5, between 2000 and 3200 (both included).

mylist = []

for num in range(2000, 3201):
    if num%7 == 0 and num%5 != 0:
        mylist.append(num)
print(mylist)

#Q2: Factorial of a given number

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(4))

# Q3:  Generate a Dictionary that contain (i, i*i)

def get_dict(n):
    mydict = {}
    for i in range(1, n + 1):
        mydict[i] = i + 1
    return mydict

print(get_dict(10))

# Q4: A non empty string and an intgral n, return a new string where the char at index n has been removed
def missing_char(str, n):
    if 0 <= n < len(str):
        return str[:n] + str[n+1:]
    else:
        print("Invalid Index")
print(missing_char('kitten', 1))
print(missing_char('kitten', 0))
print(missing_char('kitten', 4))
print(missing_char('Python', 1))

# Q5: A Numpy program to convert a Numpy array into a python list
np_array = np.array([[0,1],[2,3],[4,5]])
mylist = np_array.tolist()

print('Original Array: ',np_array )
print('List', mylist)

# Q6: A Numpy program to compute the covariance matrix of two given arrays
array1 = ([0,1,2])
array2 = ([2,1,0])

com_array = np.vstack((array1,array2))

conv_matrix = np.cov(com_array)

print("Original array 1: ", array1)
print("Original array 2: ", array2)

print("Convariance matrix of the said arrays: ", conv_matrix)

# Q7: A program that calcualtes and prints the value according to the given formula: Q = Square root of [(2 * C * D)/ H]

C = 50
H = 30

def cal_Q(D):
    return int(round(math.sqrt((2 * C * D)/H)))

input_value = input("Enter comma-seperated values of D: ")

D_values = [int(value) for value in input_value.split(',')]

result = [cal_Q(D) for D in D_values]
print(','.join(map(str, result)))