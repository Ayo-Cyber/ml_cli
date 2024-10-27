import click
from ml_cli.commands.init import init 
from ml_cli.commands.run import run

@click.group()
def cli():
    pass

# Register the init command
cli.add_command(init)
cli.add_command(run)

if __name__ == '__main__':
    cli()
