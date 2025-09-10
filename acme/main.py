
import os
import yaml
import subprocess


def run_pipeline(config_path: str = "config.yaml") -> None:
    """Runs the complete ML pipeline, from data loading to prediction.

    Args:
        config_path: Path to the configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    # For simplicity, we directly call the generated script.
    # In a real-world scenario, this would be a more robust system.
    script_path = os.path.join(output_dir, "best_model_pipeline.py")

    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Pipeline script not found at: {script_path}")

    print(f"🚀 Executing pipeline script: {script_path}")
    subprocess.run(["python3", script_path], check=True)
    print("✅ Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
