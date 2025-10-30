import pytest
import tempfile
import os
from click.testing import CliRunner
from ml_cli.cli import cli
import pandas as pd

def test_full_ml_pipeline():
    """Integration test for the complete ML pipeline: init -> eda -> preprocess -> train -> predict"""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100),
            'target': [0, 1] * 50
        })
        data_path = os.path.join(tmpdir, 'data.csv')
        data.to_csv(data_path, index=False)

        # Change to temp directory for isolated test
        with runner.isolated_filesystem(temp_dir=tmpdir):
            # Copy data to current dir
            data.to_csv('data.csv', index=False)

            # 1. Initialize project
            result = runner.invoke(cli, ['init'], input='data.csv\ntarget\noutput\n4\n')
            assert result.exit_code == 0
            assert os.path.exists('config.yaml')

            # 2. Run EDA
            result = runner.invoke(cli, ['eda'])
            assert result.exit_code == 0
            assert os.path.exists('summary_statistics.csv')

            # 3. Preprocess data
            result = runner.invoke(cli, ['preprocess'])
            assert result.exit_code == 0
            assert os.path.exists('output/preprocessed_data.csv')

            # 4. Train model
            result = runner.invoke(cli, ['train'])
            assert result.exit_code == 0
            assert os.path.exists('output/fitted_pipeline.pkl')

            # 5. Make predictions
            result = runner.invoke(cli, ['predict', '-i', 'data.csv', '-o', 'predictions.csv', '-m', 'output'])
            assert result.exit_code == 0
            assert os.path.exists('predictions.csv')
