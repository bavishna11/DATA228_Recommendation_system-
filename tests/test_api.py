from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test if the API health endpoint is returning 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"

def test_predict_sentiment_positive():
    """Test the sentiment analysis endpoint with a strongly positive review."""
    response = client.post(
        "/predict_sentiment",
        json={"text": "This product is absolutely amazing and perfect! I love it."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert data["confidence"] > 0.7

def test_predict_sentiment_negative():
    """Test the sentiment analysis endpoint with a strongly negative review."""
    response = client.post(
        "/predict_sentiment",
        json={"text": "This is terrible, broken garbage. A complete waste of money."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Negative"
    assert data["confidence"] > 0.7

def test_predict_sentiment_neutral():
    """Test the sentiment analysis endpoint with a neutral review."""
    response = client.post(
        "/predict_sentiment",
        json={"text": "The box is brown and it weighs 5 pounds."}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Neutral"
