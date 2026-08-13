import pytest

from app.backend.services.run_manager import RUN_SUBDIRECTORIES, RunManager


def test_generate_run_id_has_stable_shape(tmp_path):
    manager = RunManager(runs_dir=tmp_path)

    run_id = manager.generate_run_id()

    assert run_id
    assert "_" in run_id
    assert len(run_id.split("_")) == 2


def test_create_run_folder_creates_required_subfolders(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = manager.generate_run_id()

    paths = manager.create_run(run_id)

    assert paths.root.exists()
    for subdirectory in RUN_SUBDIRECTORIES:
        assert (paths.root / subdirectory).is_dir()


def test_run_path_traversal_is_rejected(tmp_path):
    manager = RunManager(runs_dir=tmp_path)

    with pytest.raises(ValueError):
        manager.get_paths("../outside")

    with pytest.raises(ValueError):
        manager.create_run("../outside")


def test_list_runs_skips_folders_without_metadata(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    manager.create_run("empty-run")

    assert manager.list_runs() == []
