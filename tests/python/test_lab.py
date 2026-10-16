"""Exercise actual local HTTP boundaries and checkpoint downloads."""

import json
from pathlib import Path
import sys
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest
from engram_network import WellNetwork

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from serve_lab import make_server


@pytest.fixture
def lab():
    server = make_server()
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def post(base, route, data, **headers):
    request = Request(
        base + "/api/" + route,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urlopen(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as error:
        error.read()
        error.close()
        raise


def test_live_training_probe_dream_and_checkpoint(lab):
    trained = post(lab, "train", {"episodes": 4})
    assert trained["ready"] and trained["report"]["accuracy"] > 0.9
    probe = post(lab, "probe", {"family": 2, "seed": 87})
    assert probe["predicted_family"] == "rapid orbit"
    dream = post(lab, "dream", {"seed": 5, "walk_every": 4, "dream_noise": 0.2})
    assert len(dream["itinerary"]) == 6
    with urlopen(lab + "/api/checkpoint") as response:
        assert "network.engram" in response.headers["Content-Disposition"]
        restored = WellNetwork.load(response.read())
    assert restored.memory._total_absorbed == 24
    assert restored.describe() == trained["report"]["wells"]
    with urlopen(lab + "/api/export") as response:
        state = json.load(response)
    assert state["current"]["dream"] == dream["dream"]
    with urlopen(lab) as response:
        assert b"A network made of wells" in response.read()


@pytest.mark.parametrize(
    "route,body",
    [
        ("probe", {}),
        ("dream", {}),
        ("train", {"episodes": 0}),
        ("train", {"noise": float("nan")}),
        ("train", {"seed": True}),
    ],
)
def test_invalid_controls_are_errors(lab, route, body):
    with pytest.raises(HTTPError) as caught:
        post(lab, route, body)
    assert caught.value.code == 400


def test_cross_origin_rejected_and_unknown_paths(lab):
    with pytest.raises(HTTPError) as caught:
        post(lab, "train", {}, Origin="https://example.com")
    assert caught.value.code == 403
    with pytest.raises(HTTPError) as caught:
        urlopen(lab + "/../../README.md")
    assert caught.value.code == 404
    caught.value.read()
    caught.value.close()
