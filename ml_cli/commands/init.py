import click
import json
import yaml

@click.command()
@click.option('--format', default='yaml', type=click.Choice(['yaml', 'json']), help='Format of the configuration file (yaml or json)')
def init(format):
    """Initialize a new configuration file (YAML or JSON)"""
    # Collecting user input for configuration
    data_path = click.prompt('Please enter the data directory path', type=str)
    target_column = click.prompt('Please enter the target variable column', type=str)

    # Display options for model type
    click.echo("Model type selection:")
    click.echo("1. Supervised")
    click.echo("2. Unsupervised")

    # Prompt for model type with a numeric menu
    model_type_option = click.prompt('Please select the model type (1 or 2)', type=int)

    # Map the numeric selection to the corresponding model type
    if model_type_option == 1:
        model_type = 'supervised'
    elif model_type_option == 2:
        model_type = 'unsupervised'
    else:
        raise click.BadParameter('Invalid selection. Please choose either 1 or 2.')

    # Prepare configuration data
    config_data = {
        'data': {
            'data_path': data_path,
            'target_column': target_column
        },
        'model': {
            'type': model_type,
        }
    }

    # Write the configuration file based on the specified format
    config_filename = f'config.{format}'
    if format == 'yaml':
        with open(config_filename, 'w') as config_file:
            yaml.dump(config_data, config_file)
    else:
        with open(config_filename, 'w') as config_file:
            json.dump(config_data, config_file, indent=4)

    click.echo(f"Configuration file created: {config_filename}")
