import typer
from typing import *
from script_helpers.compare_models_on_dataset import compare_models_on_dataset
from script_helpers.create_modelXdataset import create_analysis_cache
from script_helpers.preprocess import preprocess

app = typer.Typer()

app.command("compare_models")(compare_models_on_dataset)
app.command("analyze")(create_analysis_cache)
app.command("all")(preprocess)

if __name__ == "__main__":
    app()