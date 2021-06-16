# LMdiff

Comparing language models

## Setting up
From the root directory:

```
conda env create -f environment.yml
conda activate LMdiff
pip install -e .
dvc pull
```

Run the backend in development mode, deploying default models and configurations:

```
uvicorn backend.server:app --reload
```

<details>
<summary><b>For production (single worker)</b></summary>

```
uvicorn backend.server:app
```

</details>

## Comparing Language Models
Preprocess the models on a dataset before launching the app. Example:

```
python scripts/preprocess.py all gpt2-medium distilgpt2 data/datasets/glue_mrpc_1+2.csv --output-dir data/sample/gpt2-glue-comparisons
```

Then start the app:

```
python backend/server/main.py --config data/sample/gpt2-glue-comparisons
```

Note that if you instead use a `bert` based tokenization scheme, you will need to tell the frontend how to visualize the tokens:

```
python backend/server/main.py --config data/sample/bert-glue-comparisons -t bert
```

The dataset is a simple text file, with a new phrase on every line, with a bit of metadata header at the top. It can be created with:

```python
from analysis.create_dataset import create_text_dataset
import datasets
import path_fixes as pf

glue_mrpc = datasets.load_dataset("glue", "mrpc", split="train")
name = "glue_mrpc_train"

def ds2str(glue):
    """(e.g.,) Turn the first 50 sentences of the dataset into sentence information"""
    sentences = glue['sentence1'][:50]
    return "\n".join(sentences)

create_text_dataset(glue_mrpc, name, ds2str, pf.DATASETS)
```



<details>
<summary><b>Testing</b></summary>

```
make test
```

or

```
python -m pytest tests
```

All tests are stored in `tests`.

</details>

### Frontend

We like [`pnpm`](https://pnpm.io/installation) but `npm` works just as well. We also like [`Vite`](https://vitejs.dev/) for its rapid hot module reloading and pleasant dev experience. This repository uses [`Vue`](https://vuejs.org/) as a reactive framework.

From the root directory:

```
cd client
pnpm install --save-dev
pnpm run dev
```

If you want to hit the backend routes, make sure to also run the `uvicorn backend.server:app` command from the project root.

<details>
<summary><b>For production (serve with Vite)</b></summary>

```
pnpm run serve
```

</details>

<details>
<summary><b>For production (serve with this repo's FastAPI server)</b></summary>

```
cd client
pnpm run build:backend
cd ..
uvicorn backend.server:app
```

Or the `gunicorn` command from above.

All artifacts are stored in the `client/dist` directory with the appropriate basepath.
</details>

<details>
<summary><b>For production (serve with external tooling like NGINX)</b></summary>

```
pnpm run build
```

All artifacts are stored in the `client/dist` directory.
</details>

## Notes

- Check the endpoints by visiting `<localhost>:<port>/docs`