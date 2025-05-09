import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import uuid
import os

# Set page config
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide"
)

# Title and description
st.title("🌤️ Weather Prediction App")
st.write("Upload historical weather data (CSV) to train models for each city and predict tomorrow's weather.")

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}  # Dictionary to store models by location
    st.session_state.scalers = {}  # Dictionary to store scalers by location
    st.session_state.label_encoders = {}  # Dictionary to store encoders by location
    st.session_state.feature_columns = None
    st.session_state.last_data = {}  # Dictionary to store last data by location
    st.session_state.session_id = str(uuid.uuid4())

# Define target columns
TARGET_COLUMNS = ['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']

# Paths for saved model components (per location)
def get_model_paths(location):
    return {
        'model': os.path.join(MODEL_DIR, f'weather_model_{location}.pkl'),
        'scaler': os.path.join(MODEL_DIR, f'scaler_{location}.pkl'),
        'encoders': os.path.join(MODEL_DIR, f'label_encoders_{location}.pkl'),
        'features': os.path.join(MODEL_DIR, f'feature_columns_{location}.pkl'),
        'last_data': os.path.join(MODEL_DIR, f'last_data_{location}.pkl')
    }

def load_saved_models(locations):
    loaded = False
    for location in locations:
        paths = get_model_paths(location)
        if all(os.path.exists(path) for path in [paths['model'], paths['scaler'], paths['encoders'], paths['features']]):
            try:
                st.session_state.models[location] = joblib.load(paths['model'])
                st.session_state.scalers[location] = joblib.load(paths['scaler'])
                st.session_state.label_encoders[location] = joblib.load(paths['encoders'])
                st.session_state.feature_columns = joblib.load(paths['features'])
                if os.path.exists(paths['last_data']):
                    st.session_state.last_data[location] = joblib.load(paths['last_data'])
                loaded = True
            except Exception as e:
                st.error(f"Error loading saved model for {location}: {e}")
    if loaded:
        st.success("Loaded previously trained models!")
    return loaded

# Function to load and clean data
def load_and_clean_data(uploaded_file):
    with st.spinner("Loading and cleaning data..."):
        try:
            # Read the first few lines of the CSV for debugging
            uploaded_file.seek(0)
            raw_content = uploaded_file.read().decode('utf-8').splitlines()[:5]
            st.write("Raw CSV content (first 5 lines):", raw_content)
            
            # Reset file pointer and read CSV
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Debug: Show the column names read from CSV
            st.write("Columns found in CSV:", list(df.columns))
            
            # Check if columns are numeric (indicating missing header)
            expected_columns = ['Location', 'Date_Time'] + TARGET_COLUMNS
            if all(col.isdigit() for col in df.columns):
                if len(df.columns) == len(expected_columns):
                    st.warning("CSV lacks header row. Assigning expected column names.")
                    df.columns = expected_columns
                else:
                    raise ValueError(f"CSV has {len(df.columns)} columns, but expected {len(expected_columns)} columns: {expected_columns}")
            
            # Check if all expected columns are present
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"CSV is missing these columns: {missing_columns}")
            
            # Data cleaning
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            df[non_numeric_columns] = df[non_numeric_columns].fillna(df[non_numeric_columns].mode().iloc[0])
            
            # Parse Date_Time to datetime format
            df['Date_Time'] = pd.to_datetime(df['Date_Time'], errors='coerce')
            
            # Feature Engineering
            df['day_of_year'] = df['Date_Time'].dt.dayofyear
            df['month'] = df['Date_Time'].dt.month
            
            return df

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

# Function to train model for a specific location
def train_model_for_location(df, location):
    with st.spinner(f"Training model for {location}..."):
        # Filter data for the specific location
        df_location = df[df['Location'] == location].copy()
        if df_location.empty:
            st.warning(f"No data found for {location}. Skipping training.")
            return False
        
        # Prepare features and target
        X = df_location.drop(columns=TARGET_COLUMNS + ['Date_Time', 'Location'])
        y = df_location[TARGET_COLUMNS]
        
        # Store feature columns
        st.session_state.feature_columns = X.columns.tolist()

        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for column in categorical_columns:
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
        st.session_state.label_encoders[location] = label_encoders
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.session_state.scalers[location] = scaler
        
        # Train the multi-output model
        base_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model = MultiOutputRegressor(base_model)
        model.fit(X_train_scaled, y_train)
        st.session_state.models[location] = model
        
        # Store the last row for prediction
        st.session_state.last_data[location] = df_location.iloc[-1].to_dict()
        
        # Save model components
        paths = get_model_paths(location)
        joblib.dump(model, paths['model'])
        joblib.dump(scaler, paths['scaler'])
        joblib.dump(label_encoders, paths['encoders'])
        joblib.dump(st.session_state.feature_columns, paths['features'])
        joblib.dump(st.session_state.last_data[location], paths['last_data'])
        
        # Model performance
        y_pred = model.predict(X_test_scaled)
        st.subheader(f"Model Training Results for {location}")
        cols = st.columns(len(TARGET_COLUMNS))
        for i, target in enumerate(TARGET_COLUMNS):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            with cols[i]:
                st.metric(f"{target}", f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}")
        
        return True

# File uploader section
st.header("Step 1: Train Your Models or Load Existing")
if not st.session_state.models:  # Only try loading if no models are in session
    uploaded_file = st.file_uploader("Upload historical weather data CSV", type="csv", key=st.session_state.session_id)
    
    if uploaded_file is not None:
        df = load_and_clean_data(uploaded_file)
        
        if df is not None:
            # Display the first few rows of the cleaned data
            st.subheader("Preview of Your Data")
            st.dataframe(df.head())
            
            # Get unique locations
            locations = df['Location'].unique()
            
            # Try loading existing models for these locations
            if not load_saved_models(locations):
                if st.button("Train Models for All Cities"):
                    for location in locations:
                        success = train_model_for_location(df, location)
                        if success:
                            st.success(f"Model trained and saved for {location}!")
            
            # Reset button
            if st.button("Reset and Train New Models"):
                for location in locations:
                    paths = get_model_paths(location)
                    for path in paths.values():
                        if os.path.exists(path):
                            os.remove(path)
                st.session_state.models = {}
                st.session_state.scalers = {}
                st.session_state.label_encoders = {}
                st.session_state.feature_columns = None
                st.session_state.last_data = {}
                st.experimental_rerun()

# Prediction section
st.header("Step 2: Predict Tomorrow's Weather")

if st.session_state.models and st.session_state.feature_columns is not None:
    # Select city for prediction
    st.subheader("Select City for Prediction")
    selected_city = st.selectbox("Which city's weather do you want to predict?", options=list(st.session_state.models.keys()))
    
    # Display last data used for prediction
    if selected_city in st.session_state.last_data:
        st.subheader(f"Last Data Used for {selected_city}")
        last_data_df = pd.DataFrame([st.session_state.last_data[selected_city]])
        st.dataframe(last_data_df)
    
    # Create input form for prediction
    st.subheader(f"Enter Tomorrow's Weather Features for {selected_city}")
    input_data = {}
    cols = st.columns(3)
    
    for i, column in enumerate(st.session_state.feature_columns):
        col = cols[i % 3]
        if column in st.session_state.label_encoders.get(selected_city, {}):
            categories = st.session_state.label_encoders[selected_city][column].classes_
            default_value = st.session_state.last_data.get(selected_city, {}).get(column, categories[0])
            try:
                default_index = list(categories).index(default_value)
            except ValueError:
                default_index = 0
            selected = col.selectbox(f"{column}", options=categories, index=default_index)
            input_data[column] = selected
        else:
            min_val = -1000.0
            max_val = 1000.0
            default_val = float(st.session_state.last_data.get(selected_city, {}).get(column, 0.0))
            value = col.number_input(f"{column}", min_value=min_val, max_value=max_val, value=default_val)
            input_data[column] = value
    
    if st.button("Predict Tomorrow's Weather"):
        with st.spinner("Making prediction..."):
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables
                for column in st.session_state.label_encoders.get(selected_city, {}):
                    input_df[column] = st.session_state.label_encoders[selected_city][column].transform(input_df[column])
                
                # Scale features
                input_scaled = st.session_state.scalers[selected_city].transform(input_df)
                
                # Make prediction
                prediction = st.session_state.models[selected_city].predict(input_scaled)[0]
                
                # Display prediction
                st.subheader("Prediction Result")
                st.write(f"Predicted Weather for Tomorrow in {selected_city}:")
                cols = st.columns(len(TARGET_COLUMNS))
                for i, (target, value) in enumerate(zip(TARGET_COLUMNS, prediction)):
                    with cols[i]:
                        st.metric(target, f"{value:.2f}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Ensure input values match the training data format.")
else:
    st.warning("Please train models first by uploading historical data above or ensure all saved model files are available.")