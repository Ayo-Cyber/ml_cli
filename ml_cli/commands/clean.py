import click
import os
import logging
from ml_cli.utils.utils import log_artifact

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

@click.command()
def clean():
    """Clean up all generated artifacts recorded in .artifacts.log."""

    artifacts_log_path = ".artifacts.log"
    
    # Check if the artifact log file exists
    if not os.path.isfile(artifacts_log_path):
        click.secho("No artifacts log found. Nothing to clean.", fg="yellow")
        return

    # Track deleted and missing artifacts
    deleted_artifacts = []
    missing_artifacts = []

    with open(artifacts_log_path, "r") as file:
        # Read each line, which should contain a path to an artifact
        for line in file:
            artifact_path = line.strip()
            if os.path.isfile(artifact_path):
                # Delete the artifact
                os.remove(artifact_path)
                logging.info(f"Removed artifact: {artifact_path}")
                deleted_artifacts.append(artifact_path)
            else:
                # Log if the file is missing
                logging.warning(f"Artifact not found (already deleted or moved): {artifact_path}")
                missing_artifacts.append(artifact_path)

    # Cleanup the artifacts log file after removing files
    os.remove(artifacts_log_path)
    logging.info("Artifacts log cleared.")

    # Provide summary of the cleanup operation
    click.secho(f"\nCleanup Summary:", fg="cyan", bold=True)
    click.secho(f" - Total artifacts deleted: {len(deleted_artifacts)}", fg="green")
    click.secho(f" - Total artifacts missing: {len(missing_artifacts)}", fg="yellow")
    logging.info("Cleanup completed.")

# Add the clean command to the CLI entry points if necessary
