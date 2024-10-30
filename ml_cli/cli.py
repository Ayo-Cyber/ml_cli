import click
from ml_cli.commands.init import init 
from ml_cli.commands.run import run
from ml_cli.commands.eda import eda
from ml_cli.commands.preprocess import preprocess
from ml_cli.commands.clean import clean

# Custom command class to handle color and borders in help text
class CustomHelpCommand(click.Group):
    def format_help(self, ctx, formatter):
        # Title and usage section with borders
        click.secho("-" * 40, fg="blue", bold=True)
        click.secho("Usage:", fg="yellow", bold=True)
        click.echo("  ml [OPTIONS] COMMAND [ARGS]...\n")
        
        # Options section with borders
        click.secho("\n" + "-" * 40, fg="blue", bold=True)
        click.secho("Options:", fg="yellow", bold=True)
        click.echo(click.style("  --help", fg="cyan") + "  Show this message and exit.\n")
        
        # Commands section with borders
        click.secho("\n" + "-" * 40, fg="blue", bold=True)
        click.secho("Commands:", fg="yellow", bold=True)
        commands = [
            ("init", "Initialize a new configuration file (YAML or JSON)"),
            ("eda", "Perform exploratory data analysis on the dataset."),
            ("preprocess", "Preprocess the dataset to handle non-numeric columns."),
            ("run", "Run the ML pipeline based on the configuration."),
            ("clean", "Clean up all generated artifacts recorded in .artifacts.log.")
        ]
        for cmd, description in commands:
            click.echo(click.style(f"  {cmd}", fg="green", bold=True) + f"  {description}")
        click.secho("-" * 40, fg="blue", bold=True)

@click.group(cls=CustomHelpCommand)
def cli():
    """Main CLI application entry point."""
    pass

# Register the commands
cli.add_command(init)
cli.add_command(run)
cli.add_command(eda)
cli.add_command(preprocess)
cli.add_command(clean)

if __name__ == "__main__":
    cli()
