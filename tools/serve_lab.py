"""Live Python well-network lab, bound only to this computer's loopback interface."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from well_network import experiment, probe
from engram_metrics import forecast_metrics


def integer(body, key, default, low, high):
    value = body.get(key, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{key} must be an integer in {low}..{high}")
    return value


def number(body, key, default, low, high):
    value = body.get(key, default)
    if (
        type(value) not in (float, int)
        or not np.isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{key} must be finite in {low}..{high}")
    return float(value)


class Lab:
    def __init__(self):
        self.network = None
        self.report = None
        self.current = None
        self.config = None

    def state(self):
        return {
            "ready": self.network is not None,
            "config": self.config,
            "report": self.report,
            "current": self.current,
        }

    def train(self, body):
        config = {
            "seed": integer(body, "seed", 7, 0, 1000000),
            "episodes": integer(body, "episodes", 20, 1, 80),
            "noise": number(body, "noise", 0.003, 0, 0.2),
            "novelty_radius": number(body, "novelty_radius", 0.15, 0.005, 2),
        }
        network, names, report = experiment(**config)
        self.network, self.names, self.report, self.config = (
            network,
            names,
            report,
            config,
        )
        self.current = report["probe"]
        return self.state()

    def run_probe(self, body):
        if self.network is None:
            raise ValueError("Train the network first")
        self.current = probe(
            self.network,
            self.names,
            seed=integer(body, "seed", 1007, 0, 1000000),
            family=integer(body, "family", 0, 0, 2),
            noise=number(body, "noise", self.config["noise"], 0, 0.2),
            horizon=integer(body, "horizon", 24, 4, 128),
        )
        return self.current

    def dream(self, body):
        if self.network is None:
            raise ValueError("Train the network first")
        well = body.get("well") or None
        if well is not None and well not in self.network.memory._well_count:
            raise ValueError("Choose an existing well")
        walk = integer(body, "walk_every", 0, 0, 128)
        result = self.network.dream(
            self.current["context"],
            well=well,
            steps=len(self.current["actual"]),
            noise=number(body, "dream_noise", 0, 0, 3),
            rng=np.random.default_rng(integer(body, "seed", 42, 0, 1000000)),
            max_radius=0.999 if body.get("stabilize") is True else None,
            walk_every=walk or None,
        )
        self.current.update(
            {
                "dream": result.trajectory.tolist(),
                "itinerary": list(result.itinerary),
                "original_radius": result.original_radius,
                "stabilized": result.stabilized,
                "dream_config": body,
                "metrics": forecast_metrics(
                    self.current["actual"],
                    result.trajectory,
                    last_observation=self.current["context"][-1],
                ),
            }
        )
        return self.current


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(
        self, content, content_type="application/json", status=200, filename=None
    ):
        if not isinstance(content, bytes):
            content = json.dumps(content, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(content)

    def allowed(self):
        authority = f"127.0.0.1:{self.server.server_port}"
        return (
            self.headers.get("Host") == authority
            and self.headers.get("Origin", f"http://{authority}")
            == f"http://{authority}"
        )

    def do_GET(self):
        if not self.allowed():
            return self.reply({"error": "Use the local lab address"}, status=403)
        if self.path == "/":
            return self.reply(
                (ROOT / "instrument/wells.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        if self.path == "/api/state":
            return self.reply(self.server.lab.state())
        if self.path == "/api/export":
            return self.reply(self.server.lab.state(), filename="engram-session.json")
        if self.path == "/api/checkpoint" and self.server.lab.network is not None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "network.engram"
                self.server.lab.network.save(path)
                return self.reply(
                    path.read_bytes(),
                    "application/octet-stream",
                    filename="network.engram",
                )
        self.reply({"error": "Unknown lab resource"}, status=404)

    def do_POST(self):
        if not self.allowed():
            return self.reply({"error": "Local origin required"}, status=403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Request must contain 1..8192 JSON bytes")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("JSON content type required")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("JSON object required")
            methods = {
                "/api/train": self.server.lab.train,
                "/api/probe": self.server.lab.run_probe,
                "/api/dream": self.server.lab.dream,
            }
            if self.path not in methods:
                return self.reply({"error": "Unknown lab operation"}, status=404)
            self.reply(methods[self.path](body))
        except (ValueError, KeyError, FloatingPointError) as error:
            self.reply({"error": str(error)}, status=400)


def make_server(port=0):
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.lab = Lab()
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with make_server(args.port) as server:
        print(f"Engram well lab: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
