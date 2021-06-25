## Using your own models

To use your own models:

1. Create a `TextDataset` of phrases to analyze

    The dataset is a simple `.txt` file, with a new phrase on every line, and with a bit of required metadata header at the top. E.g.,
    
        ```
        ---
        checksum: 92247a369d5da32a44497be822d4a90879807a8751f5db3ff1926adbeca7ba28
        name: dataset-dummy
        type: human_created
        ---

        This is sentence 1, please analyze this.
        Every line is a new phrase to pass to the model.
        I can keep adding phrases, so long as they are short enough to pass to the model. They don't even need to be one sentence long.
        ```

        The required fields in the header:

        - `checksum` :: A unique identifier for the state of that file. It can be calculated however you wish, but it should change if anything at all changes in the contents below (e.g., two phrases are transposed, a new phase added, or a period is added after a sentence)
        - `name` :: The name of the dataset. 
        - `type` :: Either `human_created` or `machine_generated` if you want to compare on a dataset that was spit out by another model

        Each line in the contents is a new phrase to compare in the language model. A few warnings:

        - Make sure the phrases are short enough that they can be passed to the model given your memory constraints
        - The dataset is fully loaded into memory to serve to the front end, so avoid creating a text file that is too large to fit in memory.

    You can create this file in several ways:

    <details>
    <summary>From a text file</summary>
    So you have already collected all the phrases you want into a text file separated by newlines. Simply run:

    ```
    python scripts/make_dataset.py path/to/my_dataset.txt my_dataset -o folder/i/want/to/save/in
    ```
    </details>
    
    <details>
    <summary>From a python object (list of strings)</summary>
    Want to only work within python?

    ```python
    from analysis.create_dataset import create_text_dataset_from_object

    my_collection = ["Phrase 1", "My second phrase"]
    create_text_dataset_from_object(my_collection, "easy-first-dataset", "human_created", "folder/i/want/to/save/in")
    ```
    </details>
    
    <details>
    <summary>From [Huggingface Datasets](https://huggingface.co/docs/datasets/)</summary>
    It can be created from one of Huggingface's provided datasets with:

    ```python
    from analysis.create_dataset import create_text_dataset_from_hf_datasets
    import datasets
    import path_fixes as pf

    glue_mrpc = datasets.load_dataset("glue", "mrpc", split="train")
    name = "glue_mrpc_train"

    def ds2str(glue):
        """(e.g.,) Turn the first 50 sentences of the dataset into sentence information"""
        sentences = glue['sentence1'][:50]
        return "\n".join(sentences)

    create_text_dataset_from_hf_datasets(glue_mrpc, name, ds2str, ds_type="human_created", outfpath=pf.DATASETS)
    ```
    </details>



2. Choose two comparable models
    
    Two models are comparable if they:

    1. Have the exact same tokenization scheme
    2. Have the exact same vocabulary

    This allows us to do tokenwise comparisons on the model. For example, this could be:
    
    - A pretrained model and a finetuned version of it (e.g., `distilbert-base-cased` and `distilbert-base-uncased-finetuned-sst-2-english`)
    - A distilled version mimicking the original model (e.g., `bert-base-cased` and `distilbert-base-cased`)
    - Different sizes of the same model architecture (e.g., `gpt2` and `gpt2-large`)


3. Preprocess the models on the chosen dataset

    ```
    python scripts/preprocess.py all gpt2-medium distilgpt2 data/datasets/glue_mrpc_1+2.csv --output-dir data/sample/gpt2-glue-comparisons
    ```

4. Start the app

    ```
    python backend/server/main.py --config data/sample/gpt2-glue-comparisons
    ```

    Note that if you use a different tokenization scheme than the default `gpt`, you will need to tell the frontend how to visualize the tokens. For example, a `bert` based tokenization scheme:

    ```
    python backend/server/main.py --config data/sample/bert-glue-comparisons -t bert
    ```
