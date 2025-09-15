import warnings
import logging
import rich_click as click

# Suppress the torch warning from TPOT before any imports
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.*")

# Set higher log level for tpot
logging.getLogger('tpot').setLevel(logging.WARNING)

from ml_cli.commands.init import init 

from ml_cli.commands.eda import eda
from ml_cli.commands.preprocess import preprocess
from ml_cli.commands.clean import clean
from ml_cli.commands.train import train
from ml_cli.commands.predict import predict
from ml_cli.commands.serve import serve

@click.group()
def cli():
    """Main CLI application entry point."""
    pass

# Register the commands
cli.add_command(init)

cli.add_command(eda)
cli.add_command(preprocess)
cli.add_command(clean)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(serve)

if __name__ == "__main__":
    cli()
