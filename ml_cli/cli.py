import warnings
import rich_click as click
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Suppress the torch warning from TPOT before any imports
warnings.filterwarnings("ignore", message="Warning: optional dependency `torch` is not available.*")

from ml_cli.commands.init import init  # noqa: E402
from ml_cli.commands.eda import eda  # noqa: E402
from ml_cli.commands.preprocess import preprocess  # noqa: E402
from ml_cli.commands.clean import clean  # noqa: E402
from ml_cli.commands.train import train  # noqa: E402
from ml_cli.commands.predict import predict  # noqa: E402
from ml_cli.commands.serve import serve  # noqa: E402
from ml_cli.commands.completion import completion  # noqa: E402


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
cli.add_command(completion)

if __name__ == "__main__":
    cli()
