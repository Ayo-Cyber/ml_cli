import os
import click
import uvicorn
import yaml

@click.command(help="""Serve the ML model as a REST API using FastAPI.

The API automatically loads the trained model and adapts to the features used during training.
No need to manually configure features - it's completely flexible!

Usage example:
  ml serve
  ml serve --port 8080 --no-reload

API Endpoints:
  GET  /            - Welcome message and model status
  GET  /health      - Health check
  GET  /model-info  - Information about loaded model and features  
  GET  /sample-input - Get sample input format for predictions
  POST /predict     - Make predictions using the trained model
  POST /reload-model - Reload model after retraining
""")
@click.option("--host", "-h", default="127.0.0.1", help="The host to bind the server to (default: 127.0.0.1).")
@click.option("--port", "-p", default=8000, help="The port to bind the server to (default: 8000).")
@click.option("--reload/--no-reload", default=True, help="Enable or disable auto-reloading of the server on code changes (default: True).")
@click.option('--config', '-c', 'config_file', default="config.yaml",
              help="Path to the configuration file (YAML or JSON).")
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

    # Check if model files exist
    fitted_pipeline_path = os.path.join(output_dir, 'fitted_pipeline.pkl')
    pipeline_path = os.path.join(output_dir, 'best_model_pipeline.py')
    feature_info_path = os.path.join(output_dir, 'feature_info.json')
    
    if not os.path.exists(feature_info_path):
        click.secho("⚠️  Warning: No trained model found!", fg="yellow")
        click.secho("   Please run 'ml train' first to train a model.", fg="yellow")
        click.secho("   The API will start but predictions will not work until a model is available.\n", fg="yellow")
    elif os.path.exists(fitted_pipeline_path):
        click.secho("✅ Trained model found! API will be fully functional.", fg="green")
        click.secho(f"   🤖 Fitted Pipeline: {fitted_pipeline_path}", fg="blue")
        click.secho(f"   📊 Features: {feature_info_path}", fg="blue")
    elif os.path.exists(pipeline_path):
        click.secho("✅ Exported pipeline found! API will be functional.", fg="green")
        click.secho(f"   📄 Pipeline: {pipeline_path}", fg="blue")
        click.secho(f"   📊 Features: {feature_info_path}", fg="blue")
        click.secho("   ℹ️  Note: Using exported pipeline (may be slower to load)", fg="yellow")
    else:
        click.secho("⚠️  Warning: No trained model found!", fg="yellow")
        click.secho("   Please run 'ml train' first to train a model.", fg="yellow")
        click.secho("   The API will start but predictions will not work until a model is available.\n", fg="yellow")
    
    click.secho(f"🚀 Starting ML Model API at http://{host}:{port}", fg="green")
    click.secho("📚 API Documentation available at:", fg="blue")
    click.secho(f"   - Swagger UI: http://{host}:{port}/docs", fg="blue")
    click.secho(f"   - ReDoc: http://{host}:{port}/redoc", fg="blue")
    click.secho("\n🔍 Key endpoints:", fg="blue")
    click.secho(f"   - Model info: http://{host}:{port}/model-info", fg="blue")
    click.secho(f"   - Sample input: http://{host}:{port}/sample-input", fg="blue")
    click.secho(f"   - Make predictions: POST http://{host}:{port}/predict", fg="blue")
    
    os.environ["ML_CLI_CONFIG"] = config_file
    uvicorn.run("ml_cli.api.main:app", host=host, port=port, reload=reload)
