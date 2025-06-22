import json
import csv

#Following script creates the csv file from the saved experiments results in JSON format

with open("experiment_results_csv/kg_cypher.json", "r", encoding="utf-8") as file:
    data = json.load(file)

with open("experiment_results_csv/kg_cypher.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Variant", "Question", "Iteration", "Response", "Time", "ROUGE-1", "ROUGE-L",
                     "BERTScore-P", "BERTScore-R","BERTScore-F1", "Manual Score (1-4)", "Hallucination present"])

    for variant, questions in data.items():
        for question, runs in questions.items():
            for i, run in enumerate(runs, start=1):
                writer.writerow([
                    variant,
                    question,
                    i,
                    run["response"],
                    run["time"],
                    run["ROUGE-1"],
                    run["ROUGE-L"],
                    run["BERTScore-P"],
                    run["BERTScore-R"],
                    run["BERTScore-F1"],
                    "",
                    ""
                ])
