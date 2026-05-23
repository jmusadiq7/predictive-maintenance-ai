import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


TEST_DATA_PATH = "test_FD001.csv"
SENSOR_ID_COLUMNS = {"unit_nr", "time_cycles"}


def _sensor_readings(row: pd.Series) -> dict[str, float]:
    return {
        column: float(value)
        for column, value in row.items()
        if column not in SENSOR_ID_COLUMNS
    }


def _post_last_10_readings(client: TestClient, engine_id: int):
    test_df = pd.read_csv(TEST_DATA_PATH)
    engine_rows = test_df[test_df["unit_nr"] == engine_id].tail(10)
    assert len(engine_rows) == 10

    response = None
    for _, row in engine_rows.iterrows():
        response = client.post(
            "/predict",
            json={
                "engine_id": engine_id,
                "sensor_readings": _sensor_readings(row),
            },
        )

    assert response is not None
    return response


def test_health_returns_200():
    with TestClient(app) as client:
        response = client.get("/health")

    print("Health response:", response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_with_sample_row_from_test_data():
    with TestClient(app) as client:
        response = _post_last_10_readings(client, engine_id=1)

    print("Sample prediction response:", response.json())
    assert response.status_code == 200
    body = response.json()
    assert body["engine_id"] == 1
    assert body["prediction"] in {"SAFE", "DANGER"}
    assert isinstance(body["confidence"], float)
    assert "alert" in body


def test_near_failure_engine_returns_danger():
    with TestClient(app) as client:
        response = _post_last_10_readings(client, engine_id=20)

    print("Near-failure prediction response:", response.json())
    assert response.status_code == 200
    body = response.json()
    assert body["engine_id"] == 20
    assert body["prediction"] == "DANGER"
    assert body["alert"]["triggered"] is True
