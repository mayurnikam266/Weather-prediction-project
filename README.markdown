# 🌤️ Weather Prediction App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Welcome to the **Weather Prediction App**! This Streamlit-powered web app 📊 lets you predict tomorrow's weather for multiple cities, including 🌡️ temperature, 💧 humidity, 🌧️ precipitation, and 💨 wind speed, using historical weather data. Train machine learning models with ease and make predictions with a sleek, user-friendly interface!

---

## 🚀 Features

- **📂 Upload & Train**: Upload a CSV with historical weather data to train models for each city.
- **🌍 Multi-City Support**: Train separate models for different locations (e.g., London, New York).
- **📈 Model Metrics**: View RMSE and MAE for each weather variable during training.
- **🔮 Predict Weather**: Input features to forecast tomorrow's weather conditions.
- **💾 Persistent Models**: Save models, scalers, and encoders in a tidy `models/` folder.
- **🗑️ Reset Option**: Clear old models and retrain with a single click.
- **🎨 Interactive UI**: Built with Streamlit for a smooth and intuitive experience.

---

## 📋 Requirements

To run the app, you’ll need:

| Requirement        | Version       |
|--------------------|---------------|
| Python             | 3.8 or higher |
| Streamlit          | 1.36.0        |
| Pandas             | 2.2.2         |
| NumPy              | 1.26.4        |
| Scikit-learn       | 1.5.1         |
| Joblib             | 1.4.2         |

Install dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

Here’s how your project folder looks:

```
weather-prediction-app/
├── 📜 app.py                # Main Streamlit app script
├── 📋 requirements.txt      # Python dependencies
├── 📝 README.md             # This file
├── 📊 weather_data.csv      # Sample input CSV (optional)
└── 📁 models/               # Stores pickled model files
    ├── weather_model_London.pkl
    ├── scaler_London.pkl
    ├── label_encoders_London.pkl
    ├── feature_columns_London.pkl
    ├── last_data_London.pkl
    └── ...                  # Files for other cities
```

---

## 🛠️ Setup Instructions

Follow these steps to get the app running:

1. **Clone the Repository** (if hosted):
   ```bash
   git clone <repository-url>
   cd weather-prediction-app
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Your CSV**:
   - Ensure your CSV has these columns:
     - `Location` (e.g., "London")
     - `Date_Time` (e.g., "2023-01-01 12:00:00")
     - `Temperature_C` (e.g., 15.5)
     - `Humidity_pct` (e.g., 65.0)
     - `Precipitation_mm` (e.g., 2.3)
     - `Wind_Speed_kmh` (e.g., 10.5)
   - Example CSV:
     ```csv
     Location,Date_Time,Temperature_C,Humidity_pct,Precipitation_mm,Wind_Speed_kmh
     London,2023-01-01 12:00:00,10.5,70.0,0.0,15.0
     NewYork,2023-01-01 12:00:00,5.0,80.0,0.5,20.0
     ```

5. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   - Open your browser at `http://localhost:8501`.

---

## 🎮 How to Use

### Step 1: Train Models 📚
- **Upload CSV**: Use the file uploader to select your weather data CSV.
- **Train Models**: Click **"Train Models for All Cities"** to train models for each city.
  - Metrics (RMSE, MAE) for temperature, humidity, etc., will be displayed.
- **Load Existing Models**: If models exist in the `models/` folder, they’ll load automatically.
- **Reset Models**: Click **"Reset and Train New Models"** to delete old models and start fresh.

### Step 2: Predict Weather 🔮
- **Select a City**: Choose a city from the dropdown.
- **View Last Data**: See the last recorded data for context.
- **Enter Features**: Input tomorrow’s features (e.g., `day_of_year`, `month`) using the form.
- **Predict**: Click **"Predict Tomorrow's Weather"** to see forecasts for:
  - 🌡️ Temperature (°C)
  - 💧 Humidity (%)
  - 🌧️ Precipitation (mm)
  - 💨 Wind Speed (km/h)

---

## 💾 Model Storage

- **Where**: All pickled files (models, scalers, encoders, etc.) are stored in the `models/` folder to keep the project root clean.
- **Files per City**:
  - `weather_model_<city>.pkl`: Trained model
  - `scaler_<city>.pkl`: Feature scaler
  - `label_encoders_<city>.pkl`: Categorical encoders
  - `feature_columns_<city>.pkl`: Feature columns
  - `last_data_<city>.pkl`: Last data for predictions
- **Example**: For "London", files like `models/weather_model_London.pkl` are created.

---

## ❓ Troubleshooting

<details>
<summary><b>Common Issues & Fixes</b></summary>

- **CSV Errors**:
  - **Issue**: "Missing columns" or "Invalid format".
  - **Fix**: Ensure the CSV has required columns (`Location`, `Date_Time`, etc.). Check debug output in the app for column names.
- **Model Loading Fails**:
  - **Issue**: "Error loading saved model".
  - **Fix**: Delete the `models/` folder and retrain using the reset button.
- **Dependency Conflicts**:
  - **Issue**: Installation errors.
  - **Fix**: Use a fresh virtual environment and verify `requirements.txt` versions.
- **Port Conflicts**:
  - **Issue**: `localhost:8501` is in use.
  - **Fix**: Streamlit will suggest an alternative port, or stop other running Streamlit apps.

</details>

---

## 🌟 Future Enhancements

- 📊 Add charts for historical data and predictions.
- 🌦️ Support more weather variables (e.g., pressure, cloud cover).
- 🔄 Implement model versioning for tracking training runs.
- 📅 Enable multi-day or batch predictions.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details (if included).

---

## 🤝 Contributing

Have ideas or fixes? Open an issue or submit a pull request on the repository!

---

## 📬 Contact

For questions, reach out via [Your Email/Platform] or open an issue.

---

*Powered by Streamlit and Scikit-learn. Updated: May 2025* 🌍