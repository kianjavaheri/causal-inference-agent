"""API-level checks: the routes, the streamed pipeline, and the error paths."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import sample_data  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _upload(client: TestClient, df, name: str = "data.csv") -> str:
    res = client.post(
        "/upload",
        files={"file": (name, io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
    )
    assert res.status_code == 200, res.text
    return res.json()["session_id"]


def test_health_lists_every_method(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["methods"]) == {"did", "rdd", "psm", "iv", "synthetic_control"}


def test_upload_profiles_panel_structure(client: TestClient) -> None:
    res = client.post(
        "/upload",
        files={
            "file": (
                "wage.csv",
                io.BytesIO(sample_data.make_did().to_csv(index=False).encode()),
                "text/csv",
            )
        },
    )
    assert res.status_code == 200
    profile = res.json()
    assert profile["n_rows"] == 720
    assert profile["structure"]["is_panel"] is True
    assert profile["structure"]["n_units"] == 60
    assert "county_id" in profile["candidates"]["unit"]


def test_full_pipeline_over_http(client: TestClient) -> None:
    sample = sample_data.SAMPLES["rdd_scholarship"]
    sid = _upload(client, sample_data.generate("rdd_scholarship"))

    plan = client.post("/plan", json={"session_id": sid, "question": sample.question}).json()
    assert plan["chosen_method"] == "rdd"

    result = client.post("/execute", json={"session_id": sid}).json()
    est = result["estimate"]
    assert est["ci_low"] <= sample.true_effect <= est["ci_high"]

    report = client.post("/report", json={"session_id": sid}).json()
    assert report["headline"]
    assert report["confidence"] in ("high", "moderate", "low")


def test_run_streams_every_step_in_order(client: TestClient) -> None:
    sample = sample_data.SAMPLES["did_minimum_wage"]
    sid = _upload(client, sample_data.generate("did_minimum_wage"))

    with client.stream(
        "GET", "/run", params={"session_id": sid, "question": sample.question}
    ) as res:
        assert res.status_code == 200
        body = "".join(res.iter_text())

    events = [line[7:] for line in body.splitlines() if line.startswith("event: ")]
    assert events.count("step_start") == 4
    assert events.count("step_done") == 4
    assert events[-1] == "complete"
    assert "error" not in events


def test_execute_can_override_the_agents_choice(client: TestClient) -> None:
    sample = sample_data.SAMPLES["sc_tobacco"]
    sid = _upload(client, sample_data.generate("sc_tobacco"))
    plan = client.post("/plan", json={"session_id": sid, "question": sample.question}).json()
    assert plan["chosen_method"] == "synthetic_control"

    # DiD is also feasible here; the user is allowed to insist on it.
    forced = client.post("/execute", json={"session_id": sid, "method": "did"}).json()
    assert forced["method"] == "did"


def test_psm_refuses_when_a_covariate_encodes_the_treatment(client: TestClient) -> None:
    """The DiD panel's `treated_group` and `post` reproduce `treated` exactly."""
    sid = _upload(client, sample_data.make_did())
    client.post("/plan", json={"session_id": sid, "question": "Did it change employment?"})
    res = client.post("/execute", json={"session_id": sid, "method": "psm"})
    assert res.status_code == 422
    assert "perfectly" in res.json()["detail"]


def test_unknown_session_is_404(client: TestClient) -> None:
    res = client.post("/plan", json={"session_id": "nonexistent", "question": "why"})
    assert res.status_code == 404


def test_blank_question_is_rejected(client: TestClient) -> None:
    sid = _upload(client, sample_data.make_did())
    assert client.post("/plan", json={"session_id": sid, "question": "   "}).status_code == 400


def test_empty_upload_is_rejected(client: TestClient) -> None:
    res = client.post(
        "/upload", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
    )
    assert res.status_code == 400


def test_execute_before_plan_is_a_conflict(client: TestClient) -> None:
    sid = _upload(client, sample_data.make_did())
    assert client.post("/execute", json={"session_id": sid}).status_code == 409


def test_samples_round_trip(client: TestClient) -> None:
    samples = client.get("/samples").json()
    assert len(samples) == 5
    for s in samples:
        loaded = client.post(f"/samples/{s['id']}/load")
        assert loaded.status_code == 200
        csv = client.get(f"/samples/{s['id']}/download")
        assert csv.status_code == 200
        assert csv.headers["content-type"].startswith("text/csv")


def test_suggestions_are_phrased_against_real_columns(client: TestClient) -> None:
    sid = _upload(client, sample_data.generate("rdd_scholarship"))
    body = client.get(f"/suggestions/{sid}").json()
    assert body["can_run"] is True
    assert body["questions"]
    joined = " ".join(body["questions"])
    assert "college_gpa" in joined and "exam_score" in joined


def test_suggestions_explain_what_is_missing_when_nothing_works(client: TestClient) -> None:
    """A panel with no treatment column: say what is absent, not just 'no'."""
    df = sample_data.make_synthetic_control()[["state", "year", "cigarette_sales"]]
    sid = _upload(client, df, "no_treatment.csv")
    body = client.get(f"/suggestions/{sid}").json()
    assert body["can_run"] is False
    assert body["questions"] == []
    assert len(body["blockers"]) == 5
    assert any(b["missing"] for b in body["blockers"])
    assert "treatment" in body["guidance"].lower()
