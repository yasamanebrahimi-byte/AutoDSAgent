from app.tools.file_utils import load_json, save_json, write_text_atomic


def test_save_json_round_trips_dict_and_list_payloads(tmp_path):
    dict_path = tmp_path / "metadata.json"
    list_path = tmp_path / "trace.json"

    save_json(dict_path, {"run_id": "abc", "value": 1})
    save_json(list_path, [{"event": "started"}, {"event": "completed"}])

    assert load_json(dict_path) == {"run_id": "abc", "value": 1}
    assert load_json(list_path) == [{"event": "started"}, {"event": "completed"}]
    assert not list(tmp_path.glob("*.tmp"))


def test_write_text_atomic_replaces_existing_content(tmp_path):
    path = tmp_path / "report.md"

    write_text_atomic(path, "old\n")
    write_text_atomic(path, "new\n")

    assert path.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob("*.tmp"))
