import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)

X = np.random.rand(50, 1) * 100  

Y = 3.5 * X + np.random.randn(50, 1) * 20

# print(X)
# print(Y)


# plt.scatter(X, Y, color='blue', label='Data Points')
# plt.title('Random Dataset')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()

model = LinearRegression()
model.fit(X, Y)

Y_pred = model.predict(X)

# plt.figure(figsize=(8,6)) 
# plt.scatter(X, Y, color='blue', label='Data Points') 
# plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line') 
# plt.title('Linear Regression on Random Dataset')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()
# plt.grid(True)
# plt.show()

# print("Slope (Coefficient):", model.coef_[0][0])
# print("Intercept:", model.intercept_[0])

intercept = model.intercept_[0]
slope = model.coef_[0][0]

print("Slope (Coefficient):", slope)
print("Intercept:", intercept)

#y = mx + c

x = int(input("Enter a value for x to predict y: "))
y = slope * x + intercept
print(f"Predicted value of y for x={x} is: {y}")