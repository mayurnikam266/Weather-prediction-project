Weather Prediction App
A Streamlit-based web application for predicting tomorrow's weather (temperature, humidity, precipitation, and wind speed) for multiple cities using historical weather data. The app trains machine learning models (Random Forest with MultiOutputRegressor) for each city and allows users to input features to predict weather conditions.
Features

Data Upload: Upload a CSV file containing historical weather data.
Model Training: Train models for each city in the dataset, with performance metrics (RMSE, MAE) displayed.
Weather Prediction: Predict tomorrow's weather (temperature, humidity, precipitation, wind speed) for a selected city.
Persistent Models: Save and load trained models, scalers, and encoders in a models/ folder to avoid retraining.
Reset Functionality: Delete existing models and train new ones if needed.
User-Friendly Interface: Built with Streamlit for easy data upload, model training, and prediction.

Requirements

Python 3.8 or higher
Dependencies listed in requirements.txt:streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2



Project Structure
weather-prediction-app/
│
├── app.py                # Main Streamlit application script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── weather_data.csv      # Sample input CSV (optional)
└── models/               # Folder for pickled model files
    ├── weather_model_London.pkl
    ├── scaler_London.pkl
    ├── label_encoders_London.pkl
    ├── feature_columns_London.pkl
    ├── last_data_London.pkl
    └── ...               # Files for other cities

Setup Instructions

Clone the Repository (if applicable):
git clone <repository-url>
cd weather-prediction-app


Create a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Prepare Input Data:

Ensure your CSV file has the following columns:
Location: City name (e.g., "London")
Date_Time: Date and time (parseable to datetime)
Temperature_C: Temperature in Celsius
Humidity_pct: Humidity percentage
Precipitation_mm: Precipitation in millimeters
Wind_Speed_kmh: Wind speed in kilometers per hour


Place the CSV file in the project directory or upload it via the app.


Run the App:
streamlit run app.py


The app will open in your default web browser (e.g., http://localhost:8501).



Usage

Step 1: Train Models or Load Existing:

Upload your historical weather data CSV using the file uploader.
Click "Train Models for All Cities" to train models for each unique location in the CSV.
If models already exist in the models/ folder, they will be loaded automatically.
Use the "Reset and Train New Models" button to delete existing models and start fresh.


Step 2: Predict Tomorrow's Weather:

Select a city from the dropdown menu.
View the last data used for that city (displayed as a table).
Enter feature values for tomorrow (e.g., day_of_year, month) using the input fields, pre-filled with the last known values.
Click "Predict Tomorrow's Weather" to see predictions for temperature, humidity, precipitation, and wind speed.


Model Files:

Trained models, scalers, encoders, feature columns, and last data are saved as .pkl files in the models/ folder.
Example: For a city named "London", files like models/weather_model_London.pkl are created.



Input CSV Format
The CSV must include at least the following columns:

Location: String (e.g., "London", "NewYork")
Date_Time: Datetime (e.g., "2023-01-01 12:00:00")
Temperature_C: Float (e.g., 15.5)
Humidity_pct: Float (e.g., 65.0)
Precipitation_mm: Float (e.g., 2.3)
Wind_Speed_kmh: Float (e.g Reikiavik, 10.5)

Example:
Location,Date_Time,Temperature_C,Humidity_pct,Precipitation_mm,Wind_Speed_kmh
London,2023-01-01 12:00:00,10.5,70.0,0.0,15.0
London,2023-01-02 12:00:00,11.0,65.0,1.2,12.5
NewYork,2023-01-01 12:00:00,5.0,80.0,0.5,20.0

Notes

Model Storage: Pickled files are stored in the models/ folder to keep the project root clean. Ensure this folder is included when moving the project.
Error Handling: The app provides feedback for invalid CSV formats, missing columns, or prediction errors.
Performance Metrics: During training, RMSE and MAE are displayed for each target variable (temperature, humidity, etc.) per city.
Scalability: The app supports multiple cities, with separate models trained for each location.

Troubleshooting

CSV Errors: Ensure the CSV has the required columns and valid data. Check the app's debug output for column names and raw content.
Model Loading Issues: If models fail to load, delete the models/ folder and retrain using the "Reset and Train New Models" button.
Dependency Issues: Verify all packages are installed correctly using requirements.txt. Update versions if compatibility issues arise.
Port Conflicts: If localhost:8501 is in use, Streamlit will prompt you to choose another port.

Future Improvements

Add support for additional weather features (e.g., pressure, cloud cover).
Implement visualizations for historical data and predictions.
Add model versioning to track different training runs.
Support batch predictions for multiple days or cities.

License
This project is licensed under the MIT License. See the LICENSE file for details (if applicable).
Contact
For questions or contributions, please contact [Your Name/Email] or open an issue on the repository.

Built with Streamlit and Scikit-learn. Last updated: May 2025.
