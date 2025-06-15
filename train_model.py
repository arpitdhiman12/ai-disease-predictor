# 📦 Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle  # To save the model

# 📥 Load dataset from online source
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# 🧪 Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 🔀 Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train a Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 💾 Save the model to a file
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'diabetes_model.pkl'")
