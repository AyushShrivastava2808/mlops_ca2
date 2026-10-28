# MLOps Workflow

This project packages a trained machine learning model as a small FastAPI service for Titanic survival prediction. It is designed as an example MLOps workflow with a model artifact, API wrapper, containerization, and automated tests.

## Overview

The application loads a pre-trained scikit-learn model from `model.joblib` and exposes it through an HTTP API. The service accepts passenger data, predicts whether the passenger survived, and returns a JSON response.

## Project Structure

- `main.py` — FastAPI application and prediction logic
- `model.joblib` — serialized trained model
- `requirements.txt` — Python dependencies
- `Dockerfile` — container configuration for running the API
- `test/` and `tests/` — test files for validation

## Features

- Health check endpoint
- Prediction endpoint with request validation
- Container-ready deployment
- Simple ML model serving workflow

## Requirements

- Python 3.12+
- pip
- Docker (optional, for containerized execution)

## Local Setup

1. Open a terminal in this directory.
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the API:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. Open the app in a browser or use curl:

   - `http://localhost:8000/`
   - `http://localhost:8000/health`

## API Usage

### GET `/`

Returns a welcome message.

### GET `/health`

Returns service health information, including whether the model has loaded successfully.

### POST `/predict`

Send passenger data in JSON format:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Example response:

```json
{
  "prediction": 0,
  "survival": "Did not survive"
}
```

## Docker

Build the image:

```bash
docker build -t titanic-survival-api .
```

Run the container:

```bash
docker run -p 8000:8000 titanic-survival-api
```

The app will be available at `http://localhost:8000`.

## Testing

Run the test suite:

```bash
pytest
```

The tests check:

- the home route
- the health endpoint
- prediction success
- validation behavior for invalid payloads

## Notes

- The model expects Titanic-style features such as `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
- This project is a lightweight example of serving a trained ML model in a production-like API environment.
- The `test_model.py` file under `test/` appears to be a standalone script for checking the model artifact directly and may not be part of the normal API test flow.
