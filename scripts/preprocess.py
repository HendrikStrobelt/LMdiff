import typer
from typing import *
from script_helpers.compare_models_on_dataset import compare_models_on_dataset
from script_helpers.create_modelXdataset import create_analysis_results

app = typer.Typer()

app.command("compare_models")(compare_models_on_dataset)
app.command("analyze")(create_analysis_results)

@app.command("all")
def preprocess(m1: str, m2: str, dataset: str, top_k: int = 10, max_clamp_rank:int=50):
    """Create all the files needed to compare `m1` to `m2` on `dataset`. 

    m1 and m2 must be comparable (i.e., have the same tokenization)

    Args:
        m1 (str): Huggingface model name
        m2 (str): Huggingface model name
        dataset (str): Path to the dataset
        top_k (int): Number of top predicted tokens to save. Defaults to 10.
        max_clamp_rank (int): Max rank to consider for diff purposes. Defaults to 50.
    """
    #TODO Check that m1 and m2 are comparable before even creating things
    #TODO Enable user to provide output directory in which to save the files

    try:
        m1f = create_analysis_results(m1, dataset, top_k = top_k)
    except FileExistsError as e:
        typer.echo(e)
        typer.echo(f"\n\tH5 Analysis file for `{m1} x {dataset}` already exists. Continuing")
        m1f = e.details['outfname']

    try:
        m2f = create_analysis_results(m2, dataset, top_k = top_k)
    except FileExistsError as e:
        typer.echo(e)
        typer.echo(f"\n\tH5 Analysis file for `{m2} x {dataset}` already exists. Continuing")
        m2f = e.details['outfname']

    compare_f1, compare_f2 = compare_models_on_dataset(m1f, m2f, max_clamp_rank=max_clamp_rank)

    typer.echo("Done")

if __name__ == "__main__":
    app()