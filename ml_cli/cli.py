import click
from ml_cli.commands.init import init 
from ml_cli.commands.run import run
from ml_cli.commands.eda import eda
from ml_cli.commands.preprocess import preprocess

@click.group()
def cli():
    pass

# Register the init command
cli.add_command(init)
cli.add_command(run)
cli.add_command(eda)
cli.add_command(preprocess)

if __name__ == '__main__':
    cli()
