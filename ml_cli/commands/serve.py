import os
import click
import uvicorn
import yaml
import logging

@click.command(help="""Deploys the trained ML model as a RESTful API using FastAPI.
This command starts a local web server that exposes several endpoints for interacting with your model,
including making predictions, checking model status, and retrieving model information.
The API automatically loads the trained model and adapts to the features used during training,
providing a flexible and easy-to-use interface.

Examples:
  ml-cli serve
  ml-cli serve --port 8080 --no-reload
  ml-cli serve -h 0.0.0.0 -p 5000 --config my_config.json

API Endpoints:
  GET  /            - Welcome message and basic API status.
  GET  /health      - Health check endpoint to verify API availability.
  GET  /model-info  - Provides detailed information about the currently loaded model and its expected features.
  POST /predict     - Accepts new data and returns predictions from the trained model.
  POST /reload-model - Reloads the model, useful after retraining without restarting the server.
""")
@click.option("--host", "-h", default="127.0.0.1",
              help="The host IP address to bind the server to. Use '0.0.0.0' to make the server accessible externally. (Default: 127.0.0.1)")
@click.option("--port", "-p", default=8000,
              help="The port number on which the server will listen for incoming requests. (Default: 8000)")
@click.option("--reload/--no-reload", default=True,
              help="Enable or disable auto-reloading of the server when code changes are detected. Useful for development. (Default: True)")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help='The absolute or relative path to the configuration file (config.yaml or config.json) used to determine the model output directory.')
def serve(host, port, reload, config_file):
    """Serve the ML model as a REST API using FastAPI."""
    
    output_dir = "output"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            try:
                config = yaml.safe_load(f)
                output_dir = config.get('output_dir', 'output')
            except yaml.YAMLError as exc:
                click.secho(f"Error reading config file: {exc}", fg="red")
                logging.error(f"Error reading config file: {exc}")

    # Check if model files exist
    fitted_pipeline_path = os.path.join(output_dir, 'fitted_pipeline.pkl')
    pipeline_path = os.path.join(output_dir, 'best_model_pipeline.py')
    feature_info_path = os.path.join(output_dir, 'feature_info.json')
    
    if not os.path.exists(feature_info_path):
        click.secho("⚠️  Warning: No trained model found!", fg="yellow")
        logging.warning("No trained model found!")
        click.secho("   Please run 'ml train' first to train a model.", fg="yellow")
        click.secho("   The API will start but predictions will not work until a model is available.\n", fg="yellow")
    elif os.path.exists(fitted_pipeline_path):
        click.secho("✅ Trained model found! API will be fully functional.", fg="green")
        logging.info("Trained model found! API will be fully functional.")
        click.secho(f"   🤖 Fitted Pipeline: {fitted_pipeline_path}", fg="blue")
        click.secho(f"   📊 Features: {feature_info_path}", fg="blue")
    elif os.path.exists(pipeline_path):
        click.secho("✅ Exported pipeline found! API will be functional.", fg="green")
        logging.info("Exported pipeline found! API will be functional.")
        click.secho(f"   📄 Pipeline: {pipeline_path}", fg="blue")
        click.secho(f"   📊 Features: {feature_info_path}", fg="blue")
        click.secho("   ℹ️  Note: Using exported pipeline (may be slower to load)", fg="yellow")
    else:
        click.secho("⚠️  Warning: No trained model found!", fg="yellow")
        logging.warning("No trained model found!")
        click.secho("   Please run 'ml train' first to train a model.", fg="yellow")
        click.secho("   The API will start but predictions will not work until a model is available.\n", fg="yellow")
    
    click.secho(f"🚀 Starting ML Model API at http://{host}:{port}", fg="green")
    logging.info(f"Starting ML Model API at http://{host}:{port}")
    click.secho("📚 API Documentation available at:", fg="blue")
    click.secho(f"   - Swagger UI: http://{host}:{port}/docs", fg="blue")
    click.secho(f"   - ReDoc: http://{host}:{port}/redoc", fg="blue")
    click.secho("\n🔍 Key endpoints:", fg="blue")
    click.secho(f"   - Model info: http://{host}:{port}/model-info", fg="blue")
    click.secho(f"   - Make predictions: POST http://{host}:{port}/predict", fg="blue")
    
    os.environ["ML_CLI_CONFIG"] = config_file
    uvicorn.run("ml_cli.api.main:app", host=host, port=port, reload=reload)
