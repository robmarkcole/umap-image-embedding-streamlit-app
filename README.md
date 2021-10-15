# umap-image-embedding-streamlit-app
App to explore umap image embeddings for MNIST class datasets.

## UMAP
Umap depends on numba which itself uses a pinned version of numpy. This dependency limitation can be avoided by splitting the generation of embeddings and the plotting of embeddings into different envs if required. Simply use the exported `embedding.npy` file

## Dev
* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`
* `pip install jupyterlab` for dev or `streamlit run app.py` for the app