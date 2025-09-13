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

def run_server(config_file):
    runner = CliRunner()
    runner.invoke(cli, ["serve", "--config", config_file])

def test_train_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy config.yaml file
            with open("config.yaml", "w") as f:
                f.write(f"data:\n  data_path: data.csv\n  target_column: target\ntask:\n  type: classification\noutput_dir: {tmpdir}/output")

            # Create a dummy data.csv file
            data = pd.DataFrame({
                'feature1': range(100),
                'feature2': range(100),
                'target': [0, 1] * 50
            })
            os.makedirs(f'{tmpdir}/output', exist_ok=True)
            data.to_csv("data.csv", index=False)

            result = runner.invoke(cli, ["train", "--config", "config.yaml"])
            assert result.exit_code == 0
            assert os.path.exists(f"{tmpdir}/output/best_model_pipeline.py")

def test_predict_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            output_dir = os.path.join(tmpdir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Create a dummy fitted model pipeline file
            pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression())])
            X_dummy = np.array([[1,2],[3,4],[5,6],[7,8]])
            y_dummy = np.array([0,1,0,1])
            pipeline.fit(X_dummy, y_dummy)
            joblib.dump(pipeline, os.path.join(output_dir, "fitted_pipeline.pkl"))

            # Create a dummy feature_info.json file
            feature_info = {"feature_names": ["feature1", "feature2"]}
            with open(os.path.join(output_dir, "feature_info.json"), "w") as f:
                json.dump(feature_info, f)

            # Create a dummy input data file
            with open("input.csv", "w") as f:
                f.write("feature1,feature2\n1,2\n3,4")

            result = runner.invoke(cli, ["predict", "-i", "input.csv", "-o", "output.csv", "-m", output_dir])
            assert result.exit_code == 0
            assert os.path.exists("output.csv")

def test_serve_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file = os.path.join(tmpdir, "config.yaml")
        with open(config_file, "w") as f:
            f.write(f"output_dir: {tmpdir}/output")
        
        server_process = multiprocessing.Process(target=run_server, args=(config_file,))
        server_process.start()
        time.sleep(5)  # Wait for the server to start

        try:
            # Test the root endpoint
            response = requests.get("http://127.0.0.1:8000")
            assert response.status_code == 200
            assert response.json() == {"message": "Welcome to the ML-CLI API!"}

            # Test the health endpoint
            response = requests.get("http://127.0.0.1:8000/health")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

        finally:
            server_process.terminate()
            server_process.join()

def test_init_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy data file
            with open("data.csv", "w") as f:
                f.write("feature1,feature2,target\n1,2,0\n3,4,1")

            # Use input to provide answers to prompts
            result = runner.invoke(cli, ["init"], input="data.csv\nclassification\ntarget\n{tmpdir}/output\n4\n".format(tmpdir=tmpdir))
            assert result.exit_code == 0
            assert os.path.exists("config.yaml")

def test_eda_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Create a dummy config.yaml file
            with open("config.yaml", "w") as f:
                f.write("data:\n  data_path: data.csv")

            # Create a dummy data.csv file
            data = pd.DataFrame({
                'feature1': range(10),
                'feature2': range(10)
            })
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
            data = pd.DataFrame({
                'feature1': ['A', 'B', 'A', 'C'],
                'feature2': [1, 2, 3, 4]
            })
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