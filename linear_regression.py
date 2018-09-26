import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""Cost Function"""
def computeCost(X, y, theta):
	temp = np.dot(X, theta) - y 	# h(x) - y
	return np.sum(np.power(temp, 2)) / (2*m)


""" Gradient Descent
	Derivative of Cost Function of Linear Regression:
	(1/m)*X⊤(Xθ−y)
"""
def gradientDescent(X, y, theta, alpha, iterations):
	for _ in range(iterations):
		temp = np.dot(X, theta) - y
		temp = np.dot(X.T, temp)
		theta -= (alpha/m) * temp
	return theta


"""Cost Function with Multiple variables"""
def computeCostMulti(X, y, theta):
	temp = np.dot(X, theta) - y
	return np.sum(np.power(temp, 2)) / (2*m)


"""Gradient Descent with Multiple variables"""
def gradientDescentMulti(X, y, theta, alpha, iterations):
	m = len(y)
	for _ in range(iterations):
		temp = np.dot(X, theta) - y
		temp = np.dot(X.T, temp)
		theta -= (alpha/m) * temp
	return theta


data = pd.read_csv("data/ex1data1.txt", header=None)

print(data.head())

X = data.iloc[:, 0]
y = data.iloc[:, 1]
m = len(y)

plt.scatter(X, y)
plt.xlabel("Population of City in 10,000")
plt.ylabel("Profit in $10,000")
plt.show()

X = X[:, np.newaxis]
y = y[:, np.newaxis]
theta = np.zeros([2, 1])
iterations = 1500
alpha = 0.01
ones = np.ones((m, 1))
X = np.hstack((ones, X))


J = computeCost(X, y, theta)
print(J)

theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

J = computeCost(X, y, theta)
print(J)

plt.scatter(X[:, 1], y)
plt.xlabel("Population of City in 10,000")
plt.ylabel("Profit in $10,000")
plt.plot(X[:, 1], np.dot(X, theta))
plt.show()


## Linear Regression with Multiple Variable

data = pd.read_csv("data/ex1data2.txt",header = None)
print(data.head())
X = data.iloc[:,0:2]
y = data.iloc[:,2]
m = len(y)

# Feature Normalization
X = (X-np.mean(X))/np.std(X)

print(X.head())

ones = np.ones((m,1))
X = np.hstack((ones, X))
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
y = y[:,np.newaxis]

J = computeCostMulti(X, y, theta)
print(J)

theta = gradientDescentMulti(X, y, theta, alpha, iterations)
print(theta)

J = computeCostMulti(X, y, theta)
print(J)
