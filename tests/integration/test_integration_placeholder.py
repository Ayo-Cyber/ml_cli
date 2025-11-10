import pytest
import tempfile
import os
from click.testing import CliRunner
from ml_cli.cli import cli
import pandas as pd
from unittest.mock import patch


@pytest.mark.slow  # Mark as slow test for optional skipping
@patch("ml_cli.commands.init.questionary.text")
@patch("ml_cli.commands.init.questionary.select")
@patch("ml_cli.commands.init.questionary.confirm")
@patch("ml_cli.commands.init.click.confirm")
@patch("ml_cli.commands.init.click.prompt")
def test_full_ml_pipeline(mock_click_prompt, mock_click_confirm, mock_questionary_confirm, mock_select, mock_text):
    """Integration test for the complete ML pipeline: init -> eda -> preprocess -> train -> predict"""
    # Configure the mocks to return predefined answers
    mock_select.return_value.ask.side_effect = [
        "current",  # For 'Where do you want to initialize the project?'
        "classification",  # For 'Please select the task type:'
    ]
    mock_questionary_confirm.return_value.ask.return_value = True  # For 'Did you mean X?'
    mock_text.return_value.ask.return_value = "0.2"  # For test size
    
    # Mock click prompts and confirms for PyCaret config
    mock_click_confirm.side_effect = [True, True, False]  # normalize=True, feature_selection=True, remove_outliers=False
    mock_click_prompt.return_value = 1  # n_select=1 (faster for testing)

    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = pd.DataFrame({"feature1": range(100), "feature2": range(100), "target": [0, 1] * 50})
        data_path = os.path.join(tmpdir, "data.csv")
        data.to_csv(data_path, index=False)

        # Change to temp directory for isolated test
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Copy data to current dir
            data.to_csv("data.csv", index=False)

            # 1. Initialize project with PyCaret config
            # Input: data_path, target_column, output_dir (PyCaret config now via mocks)
            result = runner.invoke(cli, ["init"], input="data.csv\ntarget\noutput\n")
            assert result.exit_code == 0
            assert os.path.exists("config.yaml")

            # 2. Run EDA
            result = runner.invoke(cli, ["eda"])
            assert result.exit_code == 0
            assert os.path.exists("summary_statistics.csv")

            # 3. Preprocess data
            result = runner.invoke(cli, ["preprocess"])
            assert result.exit_code == 0
            assert os.path.exists("output/preprocessed_data.csv")

            # 4. Train model (will create pycaret_model.pkl)
            result = runner.invoke(cli, ["train"])
            assert result.exit_code == 0
            assert os.path.exists("output/pycaret_model.pkl")

            # 5. Make predictions
            result = runner.invoke(cli, ["predict", "-i", "data.csv", "-o", "predictions.csv", "-m", "output"])
            assert result.exit_code == 0
            assert os.path.exists("predictions.csv")
