# Forest Cover Type Prediction

This project predicts forest cover types based on cartographic variables using a machine learning model trained on the UCI Forest CoverType dataset.

## Features

- Backend API built with FastAPI serving prediction requests.
- Random Forest Classifier model trained with scikit-learn.
- Frontend web interface for inputting features and displaying predictions.
- Displays top feature importances dynamically after prediction.
- Reset form functionality to clear inputs and results.
- Sample data loading for quick testing.
- Model saved with compression to reduce file size.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv or virtualenv)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Forest-Cover-Type-Prediction
```

2. Create and activate a virtual environment:

```bash
python3 -m venv forest_env
source forest_env/bin/activate  # On Windows: forest_env\Scripts\activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Download and prepare the dataset:

The dataset will be downloaded and converted automatically by the training script.

5. Train the model:

```bash
python forest_cover_model.py
```

This will train the Random Forest model and save the compressed model file `forest_cover_model.pkl`.

### Running the Application

1. Start the backend server:

```bash
source forest_env/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Open `frontend.html` in a web browser.

3. Use the form to input features and get forest cover type predictions.

## Usage

- Fill in the input fields with cartographic variables.
- Click **Predict Cover Type** to get the prediction.
- Click **Load Sample Data** to fill the form with example values.
- Click **Reset Form** to clear all inputs and results.
- View the top feature importances below the prediction result.

## Notes

- The model file is compressed to reduce size for easier version control.
- The frontend communicates with the backend API at `http://localhost:8000/predict`.
- Ensure the backend server is running before using the frontend.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Covertype)
- Built with FastAPI, scikit-learn, and vanilla JavaScript.
