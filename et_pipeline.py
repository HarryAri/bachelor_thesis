import json
import torch
from sentence_transformers import util
from config.chat import chat

def load_embedded_knowledge(filepath):
    '''
    :param filepath: a path to the vector-store
    base using JSON
    :return: returns two lists: description of
    each entity and corresponding embeddings
    '''
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = []
    embeddings = []
    for item in data:
        texts.append(item["text"])
    for item in data:
        embeddings.append(item["embedding"])

    return texts, embeddings

def find_relevant_sentences(texts, embeddings, question, model):
    '''
    :param texts: description of entities
    :param embeddings: embeddings of entities description
    :param question: input query from the user
    :param model: model used to embedd the user query
    :return: the most similar semantically embedding with teh user query
    '''
    query_emb = torch.tensor(model.encode(question, convert_to_numpy=True))
    scores = util.cos_sim(query_emb, embeddings)[0]
    return texts[torch.argmax(scores)]

def run_et(texts, embeddings, question, model, llm):
    '''
    :param texts:
    :param embeddings:
    :param question: input query from the user
    :param model: model used to embedd the user query
    :param llm: the LLM used to generate the response
    :return: NPC's response to the user query
    '''
    retrieved_info = find_relevant_sentences(texts, embeddings, question, model)
    return chat(llm, question, retrieved_info)