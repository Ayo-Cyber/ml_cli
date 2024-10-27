import click
import json
import yaml
import questionary
import sys
from ml_cli.utils.utils import write_config, should_prompt_target_column, get_dependencies, is_readable_file, is_target_in_file , get_target_directory

# Define your keyboard interrupt message
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."

@click.command()
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']), help='Format of the configuration file (yaml or json)')
def init(format):
    """Initialize a new configuration file (YAML or JSON)"""

    # Determine the target directory based on user choice
    target_directory = get_target_directory()

    # Collecting user input for configuration
    data_path = click.prompt('Please enter the data directory path', type=str)

    # Check if the file path is readable
    if not is_readable_file(data_path):
        click.secho("Error: The file does not exist, is not readable, or has an unsupported format. Please provide a valid CSV, TXT, or JSON file.", fg='red')
        sys.exit(1)

    # Task type selection with questionary
    task_type = questionary.select(
        "Please select the task type:",
        choices=[
            questionary.Choice(title="Classification", value="classification"),
            questionary.Choice(title="Regression", value="regression"),
            questionary.Choice(title="Clustering", value="clustering")
        ]
    ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)

    if task_type is None:
        sys.exit(1)

    # Prompt for the target column only if necessary
    target_column = click.prompt('Please enter the target variable column', type=str) if should_prompt_target_column(task_type) else None

    # Check if the target column is present in the file
    if target_column and not is_target_in_file(data_path, target_column):
        click.secho(f"Error: The target column '{target_column}' is not present in the data file.", fg='red')
        sys.exit(1)

    # Get the dependencies based on the task type
    dependencies = get_dependencies(task_type)

    # Prepare configuration data
    config_data = {
        'data': {
            'data_path': data_path,
            'target_column': target_column,  # This will be None for clustering
        },
        'task': {
            'type': task_type,
        },
        'dependencies': dependencies,  # List dependencies here
    }

    # Use the utility function to write the configuration file in the target directory
    config_filename = os.path.join(target_directory, f'config.{format}')
    write_config(config_data, format, config_filename)

    click.echo(f"Configuration file created: {config_filename}")