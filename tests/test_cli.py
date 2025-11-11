import os
from click.testing import CliRunner
from ml_cli.cli import cli
import pandas as pd
import multiprocessing
import time
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import tempfile
import joblib
import json
import numpy as np
from unittest.mock import patch
import pytest


def run_server(config_file):
    runner = CliRunner()
    runner.invoke(cli, ["serve", "--config", config_file])


def test_train_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy config.yaml file
            with open("config.yaml", "w") as f:
                f.write(
                    f"data:\n  data_path: data.csv\n  target_column: target\ntask:\n  type: classification\noutput_dir: {tmpdir}/output\nlightautoml:\n  timeout: 60\n  cpu_limit: 2"
                )

            # Create a dummy data.csv file
            data = pd.DataFrame({"feature1": range(100), "feature2": range(100), "target": [0, 1] * 50})
            os.makedirs(f"{tmpdir}/output", exist_ok=True)
            data.to_csv("data.csv", index=False)

            result = runner.invoke(cli, ["train", "--config", "config.yaml"])
            assert result.exit_code == 0
            assert os.path.exists(f"{tmpdir}/output/lightautoml_model.pkl")


def test_predict_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Create a dummy fitted model pipeline file (simulating LightAutoML model)
            pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression())])
            X_dummy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
            y_dummy = np.array([0, 1, 0, 1])
            pipeline.fit(X_dummy, y_dummy)
            joblib.dump(pipeline, os.path.join(output_dir, "lightautoml_model.pkl"))

            # Create a dummy feature_info.json file
            feature_info = {"feature_names": ["feature1", "feature2"], "task_type": "classification"}
            with open(os.path.join(output_dir, "feature_info.json"), "w") as f:
                json.dump(feature_info, f)

            # Create a dummy input data file
            with open("input.csv", "w") as f:
                f.write("feature1,feature2\n1,2\n3,4")

            result = runner.invoke(cli, ["predict", "-i", "input.csv", "-o", "output.csv", "-m", output_dir])
            assert result.exit_code == 0
            assert os.path.exists("output.csv")


@pytest.mark.timeout(30)
def test_serve_command():
    import socket

    def wait_for_port(host, port, timeout=15.0):
        """Wait until a port starts accepting TCP connections."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=1):
                    return True
            except OSError:
                time.sleep(0.5)  # Increased from 0.2 to 0.5
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(f"output_dir: {tmpdir}/output")
        server_process = multiprocessing.Process(target=run_server, args=(config_file,))
        server_process.start()

        # Give uvicorn a moment to initialize before checking
        time.sleep(2)

        assert wait_for_port("127.0.0.1", 8000, timeout=15), "Server did not start in time"
        try:
            response = requests.get("http://127.0.0.1:8000", timeout=3)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            # API starts even without model, so status is "model_not_loaded"
            assert data["status"] in ["operational", "model_not_loaded"]

            response = requests.get("http://127.0.0.1:8000/health", timeout=3)
            assert response.status_code == 200
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] == "healthy"
        finally:
            server_process.terminate()
            server_process.join(timeout=5)
            if server_process.is_alive():
                server_process.kill()
                server_process.join()


@patch("ml_cli.commands.init.questionary.text")
@patch("ml_cli.commands.init.questionary.select")
@patch("ml_cli.commands.init.questionary.confirm")
def test_init_command(mock_confirm, mock_select, mock_text):
    # Configure the mocks to return predefined answers
    mock_select.return_value.ask.side_effect = [
        "current",  # For 'Where do you want to initialize the project?'
        "classification",  # For 'Please select the task type:'
    ]
    mock_confirm.return_value.ask.side_effect = [
        True,   # For 'Did you mean X?' (target column)
        False,  # For 'Use GPU if available?'
    ]
    mock_text.return_value.ask.side_effect = [
        "0.2",  # For test size
        "",     # For GPU IDs (empty since GPU is False)
    ]

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create sample data for testing
            data = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200), "Churn": [0, 1] * 50})
            data_file = "test_data.csv"
            data.to_csv(data_file, index=False)

            # Use input to provide answers to prompts (for click.prompt)
            # data_path, target_column, output_dir, timeout, cpu_limit
            result = runner.invoke(cli, ["init"], input=f"{data_file}\nChurn\noutput\n60\n2\n")
            assert result.exit_code == 0, f"Command failed with: {result.output}"
            assert os.path.exists("config.yaml")


def test_eda_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy config.yaml file
            with open("config.yaml", "w") as f:
                f.write("data:\n  data_path: data.csv")

            # Create a dummy data.csv file
            data = pd.DataFrame({"feature1": range(10), "feature2": range(10)})
            data.to_csv("data.csv", index=False)

            result = runner.invoke(cli, ["eda"])
            assert result.exit_code == 0
            assert os.path.exists("summary_statistics.csv")
            assert os.path.exists("eda_report.csv")
            assert os.path.exists("correlation_matrix.png")


def test_preprocess_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy config.yaml file
            with open("config.yaml", "w") as f:
                f.write(f"data:\n  data_path: data.csv\noutput_dir: {tmpdir}/output")

            # Create a dummy data.csv file with categorical data
            data = pd.DataFrame({"feature1": ["A", "B", "A", "C"], "feature2": [1, 2, 3, 4]})
            data.to_csv("data.csv", index=False)

            result = runner.invoke(cli, ["preprocess", "--config", "config.yaml"])
            assert result.exit_code == 0
            assert os.path.exists(f"{tmpdir}/output/preprocessed_data.csv")


def test_clean_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy artifact file and log
            with open("artifact.txt", "w") as f:
                f.write("dummy artifact")
            with open(".artifacts.log", "w") as f:
                f.write("artifact.txt")

            result = runner.invoke(cli, ["clean"])
            assert result.exit_code == 0
            assert not os.path.exists("artifact.txt")
            assert not os.path.exists(".artifacts.log")
