import click
from ml_cli.commands.init import init

@click.group()
def cli():
    pass

# Register the init command
cli.add_command(init)
# cli.add_command(train)

if __name__ == '__main__':
    cli()
