import click
import questionary
import sys
import os
import logging
import time
from ml_cli.utils.utils import (
    should_prompt_target_column,
    is_readable_file,
    is_target_in_file,
    get_target_directory,
    log_artifact,
    download_data,
    create_convenience_script,
    save_configuration_safely,
    get_validated_output_dir,
    get_validated_data_path_input,
)

# Constants (UI text only)
KEYBOARD_INTERRUPT_MESSAGE = "Operation cancelled by user."


@click.command(
    help="""Initialize a new configuration file (YAML or JSON).

Usage examples:
  ml init
  ml init --format json
"""
)
@click.option(
    "--format",
    default="yaml",
    type=click.Choice(["yaml", "json"]),
    help="Specify the format of the configuration file to be created (yaml or json). Default is yaml.",
)
@click.option(
    "--ssl-verify/--no-ssl-verify",
    default=True,
    help="Enable or disable SSL verification for data paths that are URLs. Default is enabled.",
)
def init(format: str, ssl_verify: bool):
    """Initialize a new configuration file (YAML or JSON)."""
    click.secho("Initializing configuration...", fg="green")

    start_time = time.time()

    try:
        # 1) Remember where we started
        original_dir = os.getcwd()

        # 2) Choose target dir (utils may chdir when selecting/creating)
        target_directory = get_target_directory()
        if target_directory is None:
            click.secho("❌ Setup cancelled.", fg="yellow")
            sys.exit(1)

        changed_directory = target_directory != original_dir
        logging.info("Target directory chosen: %s", target_directory)

        # 3) Ask for the data path (with retries handled inside util)
        data_path_input = get_validated_data_path_input(ssl_verify)
        if not data_path_input:
            click.secho("❌ Setup cancelled. Unable to get valid data path.", fg="red")
            sys.exit(1)

        logging.info("Data path provided: %s", data_path_input)

        # 4) If it's a URL, download to local project data folder; otherwise pass through
        data_path = download_data(data_path_input, ssl_verify, target_directory)
        if not data_path:
            click.secho("❌ Could not obtain data locally.", fg="red")
            sys.exit(1)

        # 5) Sanity check readability/format
        if not is_readable_file(data_path, ssl_verify=ssl_verify):
            click.secho(
                "Error: The file does not exist, is not readable, or has an unsupported format. "
                "Please provide a valid CSV, TXT, or JSON file.",
                fg="red",
            )
            logging.error("Invalid data path provided: %s", data_path)
            sys.exit(1)

        # 6) Task type
        task_type = questionary.select(
            "Please select the task type:",
            choices=[
                questionary.Choice(title="Classification", value="classification"),
                questionary.Choice(title="Regression", value="regression"),
                questionary.Choice(title="Clustering", value="clustering"),
            ],
        ).ask(kbi_msg=KEYBOARD_INTERRUPT_MESSAGE)

        if task_type is None:
            click.secho("❌ Setup cancelled.", fg="yellow")
            sys.exit(1)

        logging.info("Task type selected: %s", task_type)

        # 7) Target column (only for supervised tasks)
        target_column = None
        if should_prompt_target_column(task_type):
            for _ in range(3):
                target_column = click.prompt("Please enter the target variable column", type=str).strip()
                if target_column:
                    break
                click.secho("⚠️  Target column cannot be empty.", fg="yellow")
            if not target_column:
                click.secho("❌ No target column provided.", fg="red")
                sys.exit(1)

            target_found, corrected_target_column = is_target_in_file(data_path, target_column, ssl_verify=ssl_verify)
            if target_found:
                target_column = corrected_target_column
            else:
                click.secho(
                    f"Error: The target column '{target_column}' is not present in the data file.",
                    fg="red",
                )
                sys.exit(1)

        # 8) Test size (default-friendly and robust to cancel)
        test_size_answer = questionary.text(
            "What percentage of data should be used for testing? (e.g. 0.2 for 20%)",
            default="0.2",
        ).ask()
        if test_size_answer is None:
            test_size = 0.2
            click.secho("No input provided. Using default test size of 0.2", fg="yellow")
        else:
            try:
                test_size = float(test_size_answer)
                if not (0.1 <= test_size <= 0.5):
                    click.echo("⚠️  Warning: Test size should typically be between 0.1 and 0.5")
            except ValueError:
                test_size = 0.2
                click.echo("Invalid input, using default test size of 0.2")

        # 9) Output dir (name validity handled in util; it returns a str always)
        output_dir = get_validated_output_dir() or "output"

        # 10) TPOT generations
        generations = click.prompt("Please enter the number of TPOT generations", type=int, default=4)

        # 11) Build config
        config_data = {
            "data": {
                "data_path": data_path,
                "target_column": target_column,
            },
            "task": {"type": task_type},
            "output_dir": output_dir,
            "tpot": {"generations": generations},
            "training": {"test_size": test_size, "random_state": 42},
        }

        # 12) Persist config
        config_filename = save_configuration_safely(config_data, format, target_directory)
        if not config_filename:
            sys.exit(1)

        # 13) Log & convenience script
        log_artifact(config_filename)
        create_convenience_script(target_directory)

        elapsed_time = time.time() - start_time
        click.secho(f"Configuration file created at: {config_filename}", fg="green")
        logging.info("Configuration file created! (Time taken: %.2fs)", elapsed_time)

        # 14) Friendly wrap-up guidance
        if changed_directory:
            activate_script_path = os.path.join(target_directory, "activate.sh")
            click.secho(f"\n✅ Project initialized in: {target_directory}", fg="green", bold=True)
            click.secho(f"⚠️  Your terminal is still in: {original_dir}", fg="yellow")
            click.secho("\n💡 To move to your project directory, run:", fg="yellow")
            click.secho(f"   cd {target_directory}", fg="cyan", bold=True)
            click.secho("   # OR source the activation script:", fg="blue")
            click.secho(f"   source {activate_script_path}", fg="cyan")
        else:
            click.secho("\n✅ Project initialized in current directory!", fg="green", bold=True)
            click.secho("💡 You can now run commands like 'ml train'.", fg="yellow")

        click.secho("\n📋 Available commands:", fg="blue")
        click.secho("   ml eda        - Perform exploratory data analysis", fg="white")
        click.secho("   ml train      - Train your model", fg="white")
        click.secho("   ml serve      - Serve your model as an API", fg="white")
        click.secho("   ml predict    - Make predictions", fg="white")
        click.secho("   ml preprocess - Preprocess your data", fg="white")

        logging.info("Original directory: %s", original_dir)
        logging.info("Target directory: %s", target_directory)
        logging.info("Changed directory: %s", changed_directory)

    except KeyboardInterrupt:
        click.secho("\n❌ Operation cancelled by user.", fg="yellow")
        sys.exit(1)
    except Exception as e:
        logging.error("Unexpected error during initialization: %s", e, exc_info=True)
        click.secho(f"❌ Unexpected error: {str(e)}", fg="red")
        click.secho("Please check the logs for more details.", fg="yellow")
        sys.exit(1)
