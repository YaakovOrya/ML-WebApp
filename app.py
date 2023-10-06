import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

with st.sidebar:
    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*cG6U1qstYDijh9bPL42e-Q.jpeg")
    st.title("AutoStreamML")
    st.info("This application enables you to create and evaluate an automated Machine Learning model effortlessly. Using the power of Streamlit, "
            "Pandas, and Scikit-Learn libraries, you can upload your dataset, select features, and target variables, and the app will build a regression model for you."
            " It also provides real-time predictions based on the input values you provide."
            " Whether you're new to Machine Learning or a seasoned data scientist, our app simplifies the process of model creation and prediction.")
    st.warning("Note: Please use only numeric columns for feature selection and target variable.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.title("Model Configuration")
    numeric_columns = df.select_dtypes(include=['number']).columns
    feature_cols = st.sidebar.multiselect("Select Features", numeric_columns)
    target_col = st.sidebar.selectbox("Select Target Variable", numeric_columns)
    if feature_cols and target_col:
        st.title("Regression Model Web App")

        # Create a dictionary to store user inputs for each feature
        user_inputs = {}

        for feature in feature_cols:
            min_value = float(df[feature].min())  # Convert to float

            # Create input fields for each feature with a smaller step value
            user_inputs[feature] = st.number_input(
                f"{feature}:", min_value=min_value, value=min_value, step=0.01
            )

        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        # Display model performance
        st.write("## Model Performance")
        st.write(f"Model R^2 Score: {score:.2f}")

        st.write("## Make Predictions")
        st.write("Enter values for selected features:")

        # Use the user_inputs dictionary to get user values for prediction
        user_input_values = [user_inputs[feature] for feature in feature_cols]
        prediction = model.predict([user_input_values])
        st.write(f"Predicted {target_col}: {prediction[0]:.2f}")
