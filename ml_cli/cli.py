import warnings
import rich_click as click
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Suppress the torch warning from TPOT before any imports
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.*")

from ml_cli.commands.init import init 

from ml_cli.commands.eda import eda
from ml_cli.commands.preprocess import preprocess
from ml_cli.commands.clean import clean
from ml_cli.commands.train import train
from ml_cli.commands.predict import predict
from ml_cli.commands.serve import serve


@click.group()
def cli():
    """Main ML-CLI application entry point."""
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