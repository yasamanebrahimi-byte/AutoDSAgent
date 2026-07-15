from app.backend.services.dataset_service import generate_dataset_metadata, load_csv


def test_load_csv_and_generate_basic_metadata(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "name,age,city\n"
        "Alice,30,NY\n"
        "Bob,,LA\n"
        "Bob,,LA\n",
        encoding="utf-8",
    )

    dataframe = load_csv(csv_path)
    metadata = generate_dataset_metadata(
        dataframe=dataframe,
        filename="sample.csv",
        run_id="test-run",
    )

    assert metadata.rows == 3
    assert metadata.columns == 3
    assert metadata.column_names == ["name", "age", "city"]
    assert metadata.missing_values["age"] == 2
    assert metadata.duplicate_rows == 1
    assert len(metadata.preview) == 3
