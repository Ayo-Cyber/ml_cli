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

def run_server():
    runner = CliRunner()
    runner.invoke(cli, ["serve"])

def test_train_command():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy config.yaml file
        with open("config.yaml", "w") as f:
            f.write("data:\n  data_path: data.csv\n  target_column: target\ntask:\n  type: classification\noutput_dir: output")

        # Create a dummy data.csv file
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100),
            'target': [0, 1] * 50
        })
        os.makedirs('output', exist_ok=True)
        data.to_csv("data.csv", index=False)

        result = runner.invoke(cli, ["train"])
        assert result.exit_code == 0
        assert os.path.exists("output/best_model_pipeline.py")

def test_predict_command():
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create a dummy fitted model pipeline file
        with open("best_model_pipeline.py", "w") as f:
            f.write("""import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

tpot_pipeline = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression())])

# Fit the dummy pipeline (important for predict to work)
X_dummy = np.array([[1,2],[3,4],[5,6],[7,8]])
y_dummy = np.array([0,1,0,1])
tpot_pipeline.fit(X_dummy, y_dummy)
""")

        # Create a dummy input data file
        with open("input.csv", "w") as f:
            f.write("feature1,feature2\n1,2\n3,4")

        result = runner.invoke(cli, ["predict", "-i", "input.csv", "-o", "output.csv", "-m", "best_model_pipeline.py"])
        assert result.exit_code == 0
        assert os.path.exists("output.csv")

def test_serve_command():
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    time.sleep(5)  # Wait for the server to start

    try:
        # Test the root endpoint
        response = requests.get("http://127.0.0.1:8000")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the ML Model API"}

    finally:
        server_process.terminate()
        server_process.join()