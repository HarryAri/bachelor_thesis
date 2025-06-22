# Bachelor Thesis: Semantic World Modeling for NPC Dialogue: A Comparative Study of RAG Pipelines and Knowledge Graphs in Games

The following reporsitory contains scripts and data that was used or created due to conducted experiment using 3 different RAG pipelines.

## 1. Folder config

- chat.py: script used for generating outputs using Ollama queries

### 1.1 Folder data
This folder contains all the datasources used in different pipelines.
1. et_global_embeddings.json: data source with global scope of knowledge for vector-store base pipeline
2. et_local_embeddings.json: data source with local scope of knowledge for vector-store base pipeline
3. pt_global.txt: data source with global scope of knowledge for lexical-base pipeline
4. pt_local.txt: data source with local scope of knowledge for lexical-base pipeline
5. questions.py: contains all questions used for the experiment

