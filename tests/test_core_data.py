import pytest
import pandas as pd
import tempfile
import os
from ml_cli.core import data


def test_load_data_with_preprocessed_csv(monkeypatch):
    config = {"output_dir": "output", "data": {"data_path": "dummy.csv"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)
        preprocessed_path = os.path.join(output_dir, "preprocessed_data.csv")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(preprocessed_path, index=False)
        config["output_dir"] = output_dir
        # Should load preprocessed_data.csv
        loaded = data.load_data(config)
        assert loaded.equals(df)


def test_load_data_with_data_path(monkeypatch):
    config = {"output_dir": "output", "data": {"data_path": "dummy.csv"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "dummy.csv")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(data_path, index=False)
        config["output_dir"] = tmpdir
        config["data"]["data_path"] = data_path
        # Should load dummy.csv
        loaded = data.load_data(config)
        assert loaded.equals(df)


def test_load_data_file_not_found():
    config = {"output_dir": "output", "data": {"data_path": "notfound.csv"}}
    with pytest.raises(FileNotFoundError):
        data.load_data(config)


def test_load_data_empty_file():
    config = {"output_dir": "output", "data": {"data_path": "empty.csv"}}
    with tempfile.TemporaryDirectory() as tmpdir:
        empty_path = os.path.join(tmpdir, "empty.csv")
        open(empty_path, "w").close()
        config["output_dir"] = tmpdir
        config["data"]["data_path"] = empty_path
        with pytest.raises(pd.errors.EmptyDataError):
            data.load_data(config)
