# Bachelor Thesis: Semantic World Modeling for NPC Dialogue: A Comparative Study of RAG Pipelines and Knowledge Graphs in Games

The following reporsitory contains scripts and data that was used or created due to conducted experiment using 3 different RAG pipelines.

## Scripts description

- et_pipeline.py: this script implements RAG pipeline using SBERT + Cosine Similarity with vector-store base which finds the most semntically similar entity in vectors-tore base with user input query. The output is the response to the user input query
- pt_pipeline.py: this script implements RAG pipeline using lexical-store base with BM25 which reranks the documents and retrives top 5 of them and feeds to the LLM. The output is the response to the user input query
- kg_pipeline.py: The piepline calls neo4j and retrives all 1st degree relationship nodes from the provided knwoledge graph based on the specified NPC. Then the descriptions of the nodes are feeded into the LLM. The output is the response to the user input query
- evaluation.py: contains the function to evaluete the response using ROUGE and BertScore
- experiment.py: this script runs the experiment per pipline and knowledge scope and save it in the json format
- json_to_csv.py: this script saves all the result into csv format for further manual evaluation

### Folder "config"

- chat.py: script used for generating outputs using Ollama models
- create_embeddings.py: creates a vector-store base based on the input (JSON file)
- json_to_txt: creates a lexical-store base based on the input (JSON file)

#### Folder "data"
This folder contains all the datasources used in different pipelines.
- et_global_embeddings.json: data source with global scope of knowledge for vector-store base pipeline
- et_local_embeddings.json: data source with local scope of knowledge for vector-store base pipeline
- pt_global.txt: data source with global scope of knowledge for lexical-base pipeline
- pt_local.txt: data source with local scope of knowledge for lexical-base pipeline
- questions.py: contains all questions used for the experiment

