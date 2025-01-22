import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = 'data.csv' 
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Fill missing values in columns with the mean
df['health_limit_value'].fillna(df['health_limit_value'].mean(), inplace=True)
df['legal_limit_value'].fillna(df['legal_limit_value'].mean(), inplace=True)

# fill missing values
df['supplier_name'].fillna('Unknown', inplace=True)
df['locations_served'].fillna('Unknown', inplace=True)
df['contaminant'].fillna('Unknown', inplace=True)

st.title("🚰 Health Limit Exceedance Predictor")

st.sidebar.header("Input Data for Prediction")

# Inputs for Model Training
test_size = st.sidebar.slider("Test Size (Proportion of Test Data)", min_value=0.1, max_value=0.5, step=0.05, value=0.2)
random_state = st.sidebar.number_input("Random State (Seed for Split)", min_value=0, max_value=100, step=1, value=42)
C = st.sidebar.slider("Regularization Strength (C)", min_value=0.01, max_value=10.0, step=0.1, value=1.0)
max_iter = st.sidebar.slider("Maximum Iterations (max_iter)", min_value=100, max_value=2000, step=100, value=1000)

# Inputs for Predictions
locations = sorted(df['locations_served'].unique())
selected_location = st.sidebar.selectbox("Select Locations Served", locations)

# Filtering the dataset to include the selected location and drop rows with missing data
location_df = df[df['locations_served'] == selected_location].dropna(subset=['supplier_name', 'contaminant'])

# Filtering suppliers and contaminants for that location
suppliers_for_location = sorted(location_df['supplier_name'].unique())
contaminants_for_location = sorted(location_df['contaminant'].unique())

supplier_name = st.sidebar.selectbox("Select Supplier Name", suppliers_for_location)
contaminant = st.sidebar.selectbox("Select Contaminant", contaminants_for_location)


average_result = st.sidebar.number_input("Enter Average Result", min_value=0.0, step=0.1)
max_result = st.sidebar.number_input("Enter Maximum Result", min_value=0.0, step=0.1)
people_served = st.sidebar.number_input("Enter People Served", min_value=1, step=1)

# Label Encoding
label_encoder_contaminant = LabelEncoder()
label_encoder_supplier = LabelEncoder()
label_encoder_location = LabelEncoder()

# Fitting the LabelEncoder on the entire dataset
df['contaminant_encoded'] = label_encoder_contaminant.fit_transform(df['contaminant'])
df['supplier_name_encoded'] = label_encoder_supplier.fit_transform(df['supplier_name'])
df['locations_served_encoded'] = label_encoder_location.fit_transform(df['locations_served'])

# Feature Engineering
df['avg_to_max_ratio'] = df['average_result'] / (df['max_result'] + 1e-5) 
df['log_people_served'] = np.log1p(df['people_served'])  

# Defining features and target variable
X = df[['average_result', 'max_result', 'people_served', 'contaminant_encoded', 'avg_to_max_ratio', 
        'log_people_served', 'supplier_name_encoded', 'locations_served_encoded', 'health_limit_value', 'legal_limit_value']]
y = df['health_limit_crossed']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y)

# Training Logistic Regression Model
model = LogisticRegression(max_iter=max_iter, class_weight='balanced', C=C, solver='liblinear')
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# confusion matrix
st.subheader("📊 Model Evaluation")
cm = confusion_matrix(y_test, y_pred)
st.text("Confusion Matrix:")
st.dataframe(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

# classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Prediction for User Input
try:
    selected_data = location_df[(location_df['supplier_name'] == supplier_name) & 
                                (location_df['contaminant'] == contaminant)]
    health_limit_value = selected_data['health_limit_value'].values[0] if not selected_data.empty else None

    if health_limit_value is None:
        st.sidebar.write("❌ No health limit value found for the selected supplier and contaminant.")
    else:
        st.sidebar.write(f"Health Limit Value for selected supplier and contaminant: {health_limit_value}")

    # Transforming the input labels using the fitted label encoders
    contaminant_code = label_encoder_contaminant.transform([contaminant])[0]
    supplier_code = label_encoder_supplier.transform([supplier_name])[0]
    location_code = label_encoder_location.transform([selected_location])[0]

    user_input = pd.DataFrame([[average_result, max_result, people_served, contaminant_code, supplier_code, 
                                location_code, health_limit_value, df['legal_limit_value'].mean()]], 
                              columns=['average_result', 'max_result', 'people_served', 'contaminant_encoded', 
                                       'supplier_name_encoded', 'locations_served_encoded', 'health_limit_value', 'legal_limit_value'])

    user_input['avg_to_max_ratio'] = user_input['average_result'] / (user_input['max_result'] + 1e-5)
    user_input['log_people_served'] = np.log1p(user_input['people_served'])
    
    user_input = user_input[['average_result', 'max_result', 'people_served', 'contaminant_encoded', 'avg_to_max_ratio', 
                             'log_people_served', 'supplier_name_encoded', 'locations_served_encoded', 'health_limit_value', 'legal_limit_value']]

    # Scaling the input features
    user_input_scaled = scaler.transform(user_input)

    #prediction
    prediction = model.predict(user_input_scaled)
    result = "🔴 Health Limit Exceeded" if prediction[0] == 1 else "🟢 Health Limit Not Exceeded"

    #result
    st.subheader("📡 Prediction Result")
    st.write(f"Prediction: {result}")
    st.write(f"Input Data:\n- Location: {selected_location}\n- Supplier: {supplier_name}\n- Contaminant: {contaminant}\n- Average Result: {average_result}\n- Max Result: {max_result}\n- People Served: {people_served}")
except Exception as e:
    st.error(f"❌ Error in prediction: {e}")