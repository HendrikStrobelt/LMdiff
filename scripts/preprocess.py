import typer
import json
from pathlib import Path
from typing import *
from script_helpers.compare_models_on_dataset import compare_models_on_dataset
from script_helpers.create_modelXdataset import create_analysis_results

app = typer.Typer()

app.command("compare_models")(compare_models_on_dataset)
app.command("analyze")(create_analysis_results)

@app.command("all")
def preprocess(m1: str, m2: str, dataset: str, top_k: int = 10, max_clamp_rank:int=50, output_dir:Optional[str]=None):
    """Create all the files needed to compare `m1` to `m2` on `dataset`. 

    m1 and m2 must be comparable (i.e., have the same tokenization)

    Args:
        m1 (str): Huggingface model name
        m2 (str): Huggingface model name
        dataset (str): Path to the dataset
        top_k (int): Number of top predicted tokens to save. Defaults to 10.
        max_clamp_rank (int): Max rank to consider for diff purposes. Defaults to 50.
        output_dir (Optional[str]): 
            If provided, ignore default path configurations and create a self-contained output directory
            containing all files needed to compare two models on a dataset
    """
    #TODO Check that m1 and m2 are comparable before even creating things
    #TODO Enable user to provide output directory in which to save the files
    
    custom_folder = False
    if output_dir is not None:
        output_dir = Path(output_dir)
        assert not output_dir.is_file(), "Cannot specify an existing file."
        output_dir.mkdir(parents=True, exist_ok=True)
        typer.echo(f"Creating custom folder in {output_dir}")
        custom_folder = True

    # Create the H5 Analysis results files
    try:
        if custom_folder:
            m1f = create_analysis_results(m1, dataset, top_k = top_k, output_d = output_dir)
        else:
            m1f = create_analysis_results(m1, dataset, top_k = top_k, output_d = output_dir)
    except FileExistsError as e:
        typer.echo(e)
        typer.echo(f"\n\tH5 Analysis file for `{m1} x {dataset}` already exists. Continuing")
        m1f = e.details['outfname']

    try:
        if custom_folder:
            m2f = create_analysis_results(m2, dataset, top_k = top_k, output_d = output_dir)
        else:
            m2f = create_analysis_results(m2, dataset, top_k = top_k, output_d = output_dir)
    except FileExistsError as e:
        typer.echo(e)
        typer.echo(f"\n\tH5 Analysis file for `{m2} x {dataset}` already exists. Continuing")
        m2f = e.details['outfname']

    # Compare the files
    if custom_folder:
        compare_f1, compare_f2 = compare_models_on_dataset(m1f, m2f, max_clamp_rank=max_clamp_rank, output_dir=output_dir)
    else:
        compare_f1, compare_f2 = compare_models_on_dataset(m1f, m2f, max_clamp_rank=max_clamp_rank)

    # Save metadata, if needed
    if custom_folder:
        typer.echo("Saving metadata")
        metadata = {
            "m1": m1,
            "m2": m2,
            "dataset": dataset,
            "top_k": top_k,
            "max_clamp_rank": max_clamp_rank
        }
        with open(output_dir / "metadata.json", 'w') as fp:
            json.dump(metadata, fp)
            

    typer.echo("Done")

if __name__ == "__main__":
    app()