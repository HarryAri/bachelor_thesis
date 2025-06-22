from kg_pipeline import run_kg
from et_pipeline import run_et
from pt_pipeline import run_pt
from et_pipeline import load_embedded_knowledge
from pt_pipeline import load_plaintext
from config.data.questions import questions
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from evaluation import rouge, bertScore
import time
import json

with open("config/data/best_answers.json", "r", encoding="utf-8") as f:
    best_answers = json.load(f)

num_iterations = 3
bert_scorer = BERTScorer(model_type='bert-base-uncased', lang='en')
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def experiment_kg(llm):
    '''
    :param llm: the LLM used to generate the response
    :return: an array of results per question each containing the dictionary of finale evaluation metrics scores
    '''
    result = {}
    for question in questions:
        result[question] = []

    for i in range(num_iterations):
        print(f"KG - Iteration {i + 1}/{num_iterations}")
        for question in questions:
            start_time = time.time()
            response = run_kg(question, llm)
            elapsed = time.time() - start_time
            best_answer = best_answers[question]
            rouge1, rougeL = rouge(response, best_answer)
            P, R, F1 = bertScore(response, best_answer, bert_scorer)
            result[question].append({
                "response": response,
                "time": elapsed,
                "ROUGE-1": rouge1,
                "ROUGE-L": rougeL,
                "BERTScore-P": P,
                "BERTScore-R": R,
                "BERTScore-F1": F1,
                "Manual Score (1-4)": "",
                "Hallucination present": ""
            })

    return result

def experiment_et(path, llm):
    '''
    :param path: the path of the vector-store base
    :param llm: the LLM used to generate the response
    :return: an array of results per question each containing the dictionary of finale evaluation metrics scores
    '''

    result = {}
    texts, embeddings = load_embedded_knowledge(path)
    for question in questions:
        result[question] = []

    for i in range(num_iterations):
        print(f"ET - Iteration {i + 1}/{num_iterations}")
        for question in questions:
            start_time = time.time()
            response = run_et(texts, embeddings, question, model, llm)
            elapsed = time.time() - start_time
            best_answer = best_answers[question]
            rouge1, rougeL = rouge(response, best_answer)
            P, R, F1 = bertScore(response, best_answer, bert_scorer)
            result[question].append({
                "response": response,
                "time": elapsed,
                "ROUGE-1": rouge1,
                "ROUGE-L": rougeL,
                "BERTScore-P": P,
                "BERTScore-R": R,
                "BERTScore-F1": F1,
                "Manual Score (1-4)": "",
                "Hallucination present": ""
            })

    return result

def experiment_pt(path, llm):
    '''
    :param path: the path of the lexical datastore
    :param llm: the LLM used to generate the response
    :return: an array of results per question each containing the dictionary of finale evaluation metrics scores
    '''
    result = {}
    source = load_plaintext(path)
    model = BM25Okapi([line.lower().split() for line in source])
    print(model)
    for question in questions:
        result[question] = []

    for i in range(num_iterations):
        print(f"PT - Iteration {i + 1}/{num_iterations}")
        for question in questions:
            start_time = time.time()
            response = run_pt(source, question, model, llm)
            elapsed = time.time() - start_time
            best_answer = best_answers[question]
            rouge1, rougeL = rouge(response, best_answer)
            P, R, F1 = bertScore(response, best_answer, bert_scorer)
            result[question].append({
                "response": response,
                "time": elapsed,
                "ROUGE-1": rouge1,
                "ROUGE-L": rougeL,
                "BERTScore-P": P,
                "BERTScore-R": R,
                "BERTScore-F1": F1,
                "Manual Score (1-4)": "",
                "Hallucination present": ""
            })

    return result

def save_results(results, name):
    '''
    :param results: the results of experiment
    :param name: the way you want to name document
    :return: JSON file containg results
    '''
    with open(f"{name}.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

results_llama3_2 = {
    "pt_local": experiment_pt("config/data/pt_local.txt", "llama3.2"),
}

