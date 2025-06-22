from sentence_transformers import SentenceTransformer
import json

def preprocess_et_json(input_path, output_path):
    '''
    :param input_path: the path of the JSON file prepared for vector-store base setup
    :param output_path: the path where the vector-store base using JSON will stored
    :return: vector-store base
    '''
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def get_text(entry):
        name = entry.get("name")
        desc = entry.get("description")
        return f"{name}: {desc}".strip()

    sentences = [get_text(item) for item in data]
    embeddings = model.encode(sentences)

    output = [{"text": sent, "embedding": emb.tolist()} for sent, emb in zip(sentences, embeddings)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

preprocess_et_json("data/global.json", "data/et_global_embeddings.json")
preprocess_et_json("data/local.json", "data/et_local_embeddings.json")
