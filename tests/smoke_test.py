"""Smoke tests for core API endpoints."""


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid(client, api_headers, valid_customer):
    response = client.post("/predict", json=valid_customer, headers=api_headers)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["churn_prediction"], bool)
    assert 0 <= body["churn_probability"] <= 1


def test_predict_missing_field(client, api_headers, valid_customer):
    payload = valid_customer.copy()
    del payload["tenure"]
    response = client.post("/predict", json=payload, headers=api_headers)
    assert response.status_code == 422


def test_predictions_list(client, api_headers):
    response = client.get("/predictions", headers=api_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Telecom Churn Prediction" in response.text
