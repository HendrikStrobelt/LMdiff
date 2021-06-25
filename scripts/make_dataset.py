import typer
from pathlib import Path
from typing import *
from analysis.create_dataset import create_text_dataset_from_file

app = typer.Typer()

@app.command()
def main(fname:str, name:str, ds_type:str="human_created", outfpath:Optional[str]=typer.Option(None, "-o")):
    typer.echo(f"Opening file {fname}.")
    create_text_dataset_from_file(fname, name, ds_type=ds_type, outfpath=outfpath)
    typer.echo("Done.")


if __name__ == "__main__":
    app()