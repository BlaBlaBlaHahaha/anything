import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
df = pd.read_csv('titanic (2) (2).csv')

# Data preprocessing
df.dropna(inplace=True)
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

# Split the dataset into features and target variable
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app
def main():
    st.title("Titanic Survival Prediction")
    
    # Display accuracy
    st.write(f"Model Accuracy: {accuracy}")
    
    st.sidebar.title("Predict the survival")
    
    # Display form for user input
    pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
    sex = st.sidebar.radio("Sex", ['Female', 'Male'])
    age = st.sidebar.slider("Age", 0, 100, 25)
    sib_sp = st.sidebar.slider("Number of Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.sidebar.slider("Number of Parents/Children Aboard", 0, 6, 0)
    
    # Map user input to match dataset format
    sex = 0 if sex == 'Female' else 1
    
    # Make a prediction
    survival_probability = model.predict_proba([[pclass, sex, age, sib_sp, parch]])[0][1]
    
    # Display prediction
    st.sidebar.write(f"Survival Probability: {survival_probability}")
    
if __name__ == '__main__':
    main()