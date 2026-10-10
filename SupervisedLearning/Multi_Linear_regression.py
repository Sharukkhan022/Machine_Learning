import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# create a sample dataset

data = {
    "Hours_Studied": [2, 3, 4, 5, 6, 7],
    "Hours_Sleep":   [6, 6, 7, 7, 8, 8],
    "Marks":         [50, 55, 65, 70, 80, 85]
}

df = pd.DataFrame(data)
# print(df)

X = df[["Hours_Studied", "Hours_Sleep"]]
y = df["Marks"]

# plt.figure(figsize=(10, 5))
# plt.scatter(df["Hours_Studied"], df["Marks"], color='blue', label='Hours Studied vs Marks')
# plt.scatter(df["Hours_Sleep"], df["Marks"], color='green', label='Hours Sleep vs Marks')
# plt.title('Marks vs Hours Studied and Sleep')
# plt.xlabel('Hours')
# plt.ylabel('Marks')
# plt.legend()  
# plt.savefig("marks_vs_hours.png", dpi=150) 


model = LinearRegression()
model.fit(X, y)

print(model.coef_)
print(model.intercept_) 
predicted_marks = model.predict([[6, 7]])
print(f"Predicted marks for 6 hours studied and 7 hours sleep: {predicted_marks[0]}")

