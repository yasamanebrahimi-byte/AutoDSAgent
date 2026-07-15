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
