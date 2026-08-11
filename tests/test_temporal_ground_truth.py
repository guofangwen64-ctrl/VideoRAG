from medhorizon_videorag.datasets import MedHorizonDataset, recover_evidence


def test_recovers_direct_and_phase_anchor(tmp_path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        '{"key":"v1","video_path":"v1.mp4","qa":['
        '{"uid":1,"question":"What happens from 1:02 to 2:02?","answer":"A","options":["A. Suturing"],"task_name":"Action Recognition"},'
        '{"uid":2,"question":"At the Suturing phase onset, what instrument is visible?","answer":"A","options":["A. Needle"],"task_name":"Phase-Instrument Association"}'
        ']}\n', encoding="utf-8"
    )
    evidence = {item.qa_uid: item for item in recover_evidence(MedHorizonDataset(path))}
    assert evidence[1].method == "direct_range"
    assert evidence[1].windows == [(62.0, 122.0)]
    assert evidence[2].method == "phase_anchor"
    assert evidence[2].confidence == "weak"
