from fastapi.testclient import TestClient

from app.backend.main import app


def test_fastapi_app_imports_and_health_endpoint_works():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "autods-agent-backend",
    }
