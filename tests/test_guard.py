from moelab import guard


def test_verdict_self_calibrates_and_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "PATH", str(tmp_path / "p.json"))
    assert not guard.verdict([200.0, 199.0])["contended"]       # first run sets the record
    assert guard.verdict([150.0])["contended"]                  # 75% of record
    assert guard.verdict([250.0])["probe_best_gbs"] == 250.0    # record only goes up
    assert guard.verdict([200.0])["contended"]                  # 80% of new record


def test_probe_is_positive():
    assert guard.probe(1 << 24) > 1.0
