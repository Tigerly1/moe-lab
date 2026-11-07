from moelab import guard


def test_verdict_uses_recent_p90(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "PATH", str(tmp_path / "p.json"))
    monkeypatch.setattr(guard, "key", lambda: "chip|ac")
    assert guard.verdict([230.0, 231.0])["status"] == "calibrating"
    for _ in range(3):
        guard.verdict([235.0, 236.0])
    assert guard.verdict([232.0])["status"] == "ok"
    v = guard.verdict([190.0])                               # ~81% of p90
    assert v["status"] == "contended" and v["contended"]
    for _ in range(30):                                      # history is a window: a slower regime becomes the norm
        guard.verdict([200.0, 201.0])
    assert guard.verdict([199.0])["status"] == "ok"


def test_legacy_float_record_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "p.json"
    p.write_text('{"chip|ac": 242.0}')
    monkeypatch.setattr(guard, "PATH", str(p))
    monkeypatch.setattr(guard, "key", lambda: "chip|ac")
    assert guard.verdict([200.0])["status"] == "calibrating"


def test_probe_is_positive():
    assert guard.probe(1 << 24) > 1.0
