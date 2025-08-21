"""
Organize trained models into proper directory structure
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import json
from pathlib import Path
from config.base_config import DATA_DIR


def organize_models():
    """Organize models from results/ into models/ directory"""

    results_dir = DATA_DIR / "results"
    models_dir = DATA_DIR / "models"

    if not results_dir.exists():
        print("No results directory found.")
        return

    print("📦 Organizing trained models...")

    # Create model metadata
    model_metadata = {
        "models": [],
        "last_updated": str(pd.Timestamp.now()),
        "total_models": 0
    }

    for experiment_dir in results_dir.iterdir():
        if experiment_dir.is_dir():
            print(f"\n📁 Processing experiment: {experiment_dir.name}")

            # Look for model files
            model_files = list(experiment_dir.glob("*.pth"))

            for model_file in model_files:
                # Determine destination
                if "final_model" in model_file.name or "safe" in model_file.name:
                    dest_dir = models_dir / "production"
                    model_name = f"{experiment_dir.name}.pth"
                else:
                    dest_dir = models_dir / "experiments" / experiment_dir.name
                    model_name = model_file.name

                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / model_name

                # Copy model
                shutil.copy2(model_file, dest_path)
                print(f"  ✅ {model_file.name} → {dest_path.relative_to(DATA_DIR)}")

                # Add to metadata
                model_info = {
                    "name": model_name,
                    "path": str(dest_path.relative_to(DATA_DIR)),
                    "experiment": experiment_dir.name,
                    "type": "production" if "final_model" in model_file.name else "experiment",
                    "size_mb": round(model_file.stat().st_size / 1024 / 1024, 2)
                }

                # Try to load config for more details
                config_file = experiment_dir / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        model_info.update({
                            "num_goals": config.get("environment", {}).get("num_goals", 4),
                            "batch_size": config.get("training_config", {}).get("batch_size", 4800),
                            "learning_rate": config.get("training_config", {}).get("learning_rate", 0.01)
                        })

                model_metadata["models"].append(model_info)

    # Save metadata
    model_metadata["total_models"] = len(model_metadata["models"])

    with open(models_dir / "model_metadata.json", 'w') as f:
        json.dump(model_metadata, f, indent=2)

    print(f"\n✅ Organized {model_metadata['total_models']} models")
    print(f"📄 Model metadata saved to: {models_dir / 'model_metadata.json'}")


if __name__ == "__main__":
    import pandas as pd

    organize_models()