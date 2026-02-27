from fastapi.testclient import TestClient

from app.server import app, initialize_caches


def _client() -> TestClient:
    initialize_caches()
    return TestClient(app)


def test_mode_isolated_send_accept_flow() -> None:
    client = _client()

    credit_send = client.post('/send-offer/M-FASHION-T1?mode=grab_credit')
    assert credit_send.status_code == 200
    assert credit_send.json()["status"] == "sent"

    insurance_accept_before_send = client.post('/accept-offer/M-FASHION-T1?mode=grab_insurance')
    assert insurance_accept_before_send.status_code == 409

    insurance_send = client.post('/send-offer/M-FASHION-T1?mode=grab_insurance')
    assert insurance_send.status_code == 200

    insurance_accept = client.post('/accept-offer/M-FASHION-T1?mode=grab_insurance')
    assert insurance_accept.status_code == 200
    assert insurance_accept.json()["status"] == "mandate_initiated"


def test_rejected_merchant_cannot_receive_offer() -> None:
    client = _client()
    response = client.post('/send-offer/M-HOME-REJ?mode=grab_credit')
    assert response.status_code == 400


def test_health_and_trail_endpoints() -> None:
    client = _client()
    health = client.get('/health')
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    trail = client.get('/decision-trail/M-FASHION-T1?mode=grab_credit')
    assert trail.status_code == 200
    payload = trail.json()
    assert "primary_risk_drivers" in payload
    assert "offer" in payload
