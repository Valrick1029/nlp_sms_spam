import pytest
from fastapi.testclient import TestClient
from src.main import app
from src.pipeline import FastTextSpamPipeline

client = TestClient(app)

def test_preprocess_logic():
    pipeline = FastTextSpamPipeline("model/fasttext_spam_model.bin")
    raw_text = "Hello!!!\nThis is a test."
    clean_text = pipeline._preprocess(raw_text)
    assert "!!!" not in clean_text
    assert "\n" not in clean_text
    assert clean_text == "hello this is a test"

def test_prediction_format():
    pipeline = FastTextSpamPipeline("model/fasttext_spam_model.bin")
    result = pipeline.predict("Win money now")
    assert "label" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1

def test_api_predict_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Free entry in 2 a weekly comp to win FA Cup final tkts 21st May 2005."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["label"] in ["spam", "ham"]

def test_api_empty_text():
    response = client.post("/predict", json={"text": ""})
    # HTTPException(400) in main.py
    assert response.status_code == 400