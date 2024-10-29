import click
from ml_cli.commands.init import init 
from ml_cli.commands.run import run
from ml_cli.commands.eda import eda

@click.group()
def cli():
    pass

# Register the init command
cli.add_command(init)
cli.add_command(run)
cli.add_command(eda)

if __name__ == '__main__':
    cli()
