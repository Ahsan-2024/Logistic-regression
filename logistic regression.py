import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Data: Study hours and pass/fail (0 = fail, 1 = pass)
Hours = np.array([[1], [2], [3], [4], [5], [6], [7]])
Check = np.array([0, 0, 0, 0, 0, 1, 1])

# Train-test split
Hours_train, Hours_test, Check_train, Check_test = train_test_split(
    Hours, Check, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(Hours_train, Check_train)

# User input
user_input = int(input("Enter your Study hours for checking: "))
prediction = model.predict([[user_input]])[0]

# Show prediction result
if prediction == 1:
    print("🎉 Congratulations! You will PASS. Well done!")
else:
    print("📘 Better luck next time. Keep studying!")
# 📊 Plot the data
plt.scatter(Hours, Check, color='blue', label='Actual Data')  # Actual data points
plt.plot(Hours, model.predict(Hours), color='red', label='Prediction Line')  # Model prediction line
plt.scatter([[user_input]], [prediction], color='green', label='Your Prediction', s=100)  # User prediction
plt.xlabel('Study Hours')
plt.ylabel('Pass (1) / Fail (0)')
plt.title('Logistic Regression: Study Hours vs Result')
plt.legend()
plt.grid(True)
plt.show()    