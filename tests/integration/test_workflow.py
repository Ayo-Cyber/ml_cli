import pytest
import os
from click.testing import CliRunner
from ml_cli.cli import cli
import pandas as pd

class TestFullWorkflow:
    """Integration tests for complete ML workflow"""
    
    @pytest.mark.skip(reason="Init command requires interactive prompts - needs non-interactive mode")
    def test_complete_workflow(self, temp_project_dir, sample_csv):
        """Test complete workflow: init -> eda -> preprocess -> train"""
        runner = CliRunner()
        
        # 1. Initialize project
        result = runner.invoke(cli, ['init', 'test_ml_project', '--force'])
        assert result.exit_code == 0
        assert os.path.exists('test_ml_project')
        
        os.chdir('test_ml_project')
        
        # 2. Copy sample data
        import shutil
        shutil.copy(sample_csv, 'data/raw/train.csv')
        
        # 3. Run EDA
        result = runner.invoke(cli, ['eda', 'data/raw/train.csv'])
        assert result.exit_code == 0
        assert os.path.exists('reports/eda_report.html')
        
        # 4. Preprocess data
        result = runner.invoke(cli, [
            'preprocess',
            'data/raw/train.csv',
            '--target', 'target',
            '--test-size', '0.2'
        ])
        assert result.exit_code == 0
        assert os.path.exists('data/processed/X_train.csv')
        assert os.path.exists('data/processed/X_test.csv')
        
        # 5. Train model (quick test)
        result = runner.invoke(cli, [
            'train',
            '--data-dir', 'data/processed',
            '--target', 'target',
            '--generations', '2',
            '--population-size', '5',
            '--cv', '2'
        ])
        assert result.exit_code == 0
        assert os.path.exists('models/best_model.pkl')
    
    @pytest.mark.skip(reason="Init command requires interactive prompts - needs non-interactive mode")
    def test_predict_workflow(self, temp_project_dir, sample_csv):
        """Test prediction workflow"""
        runner = CliRunner()
        
        # Initialize and prepare
        runner.invoke(cli, ['init', 'pred_project', '--force'])
        os.chdir('pred_project')
        
        import shutil
        shutil.copy(sample_csv, 'data/raw/train.csv')
        
        # Preprocess
        runner.invoke(cli, [
            'preprocess',
            'data/raw/train.csv',
            '--target', 'target',
            '--test-size', '0.2'
        ])
        
        # Quick train
        runner.invoke(cli, [
            'train',
            '--data-dir', 'data/processed',
            '--target', 'target',
            '--generations', '1',
            '--population-size', '3',
            '--cv', '2'
        ])
        
        # Predict
        result = runner.invoke(cli, [
            'predict',
            '--model', 'models/best_model.pkl',
            '--data', 'data/processed/X_test.csv',
            '--output', 'predictions.csv'
        ])
        
        assert result.exit_code == 0
        assert os.path.exists('predictions.csv')

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_eda_nonexistent_file(self, temp_project_dir):
        """Test EDA with non-existent file"""
        runner = CliRunner()
        result = runner.invoke(cli, ['eda', 'nonexistent.csv'])
        assert result.exit_code != 0
    
    def test_train_without_data(self, temp_project_dir):
        """Test training without data"""
        runner = CliRunner()
        result = runner.invoke(cli, ['train', '--data-dir', 'nonexistent'])
        assert result.exit_code != 0
    
    def test_predict_without_model(self, temp_project_dir, sample_csv):
        """Test prediction without model"""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'predict',
            '--model', 'nonexistent.pkl',
            '--data', sample_csv
        ])
        assert result.exit_code != 0
