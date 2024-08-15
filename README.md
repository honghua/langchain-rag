# RAG demo with Llama3

## Prerequisite
You'll need to set up an Huggingface account to download LLama model from transformers


## Install dependencies

1. Now run this command to install dependenies in the `requirements.txt` file. 

    ```python
    pip install -r requirements.txt
    ```

1. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

Alice meets the Mad Hatter through the Cat,
who initially introduces her to both the Hatter and the March Hare.
The Cat tells Alice that the Hatter lives in one direction and the March Hare lives in another,
and invites her to visit either of them, as they are both mad.

```python
python query_data.py "Where is 2024 Olympics hosted?"
```

The 2024 Summer Olympics will be held in Paris, France, as well as a few other cities in France.


```python
python query_data.py "What is Boson AI"
```

Based on the provided context, Boson AI appears to be a company working on developing intelligent agents that can serve as human companions and helpers. They are specifically focused on creating advanced language models, such as Higgs-Llama-3-70B-v2, which is a new model designed to improve upon its predecessor and narrow the gap to the best proprietary models in relevant benchmarks.
Response: Based on the provided context, Boson AI appears to be a company working on developing intelligent agents that can serve as human companions and helpers. They are specifically focused on creating advanced language models, such as Higgs-Llama-3-70B-v2, which is a new model designed to improve upon its predecessor and narrow the gap to the best proprietary models in relevant benchmarks.

