import click
import uvicorn

@click.command(help="""Serve the ML model as a REST API using FastAPI.

Usage example:
  ml serve
  ml serve --port 8080 --no-reload
""")
@click.option("--host", "-h", default="127.0.0.1", help="The host to bind the server to (default: 127.0.0.1).")
@click.option("--port", "-p", default=8000, help="The port to bind the server to (default: 8000).")
@click.option("--reload/--no-reload", default=True, help="Enable or disable auto-reloading of the server on code changes (default: True).")
def serve(host, port, reload):
    """Serve the ML model as a REST API using FastAPI."""
    click.secho(f"Serving the ML model at http://{host}:{port}", fg="green")
    uvicorn.run("ml_cli.api.main:app", host=host, port=port, reload=reload)
