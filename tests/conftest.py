import pytest
import os
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    yield temp_dir
    
    os.chdir(original_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def sample_csv(temp_project_dir):
    """Create a sample CSV file for testing"""
    import pandas as pd
    
    # Create more realistic data with enough samples
    data = {
        'feature1': list(range(1, 101)),
        'feature2': list(range(100, 0, -1)),
        'feature3': (['A', 'B'] * 50),
        'target': ([0, 1] * 50)
    }
    df = pd.DataFrame(data)
    csv_path = os.path.join(temp_project_dir, 'test_data.csv')
    df.to_csv(csv_path, index=False)
    
    return csv_path

@pytest.fixture
def initialized_project(temp_project_dir):
    """Create an initialized ML project"""
    from ml_cli.commands.init import init
    from click.testing import CliRunner
    
    runner = CliRunner()
    result = runner.invoke(init, ['test_project', '--force'])
    
    if result.exit_code != 0:
        print(f"Init failed with: {result.output}")
    
    project_path = os.path.join(temp_project_dir, 'test_project')
    return project_path
