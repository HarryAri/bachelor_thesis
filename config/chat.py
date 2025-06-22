import ollama

def chat(model_name, question, retrieved_info):
    '''
    :param model_name: LLM that is being used for generating the response
    :param question: input query from the user
    :param retrieved_info: the information retrieved from data source
    :return: NPC's response to the user query
    '''
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "system", "content": f"""You are Billy, a merchant who travels between towns Shire and Nexus.
                    Give an answer to the Player question based on the given Context. Do not make up facts. 
                    You do not have a modern days knowledge.
                    Answer in 1-2 sentences. If the Context doesn't include the answer, say: 'I don’t know about that.'"""},
                  {"role": "user", "content": f"Context: {retrieved_info} \n\nPlayer Question: {question}"}],
        stream=False
    )

    return response["message"]["content"].strip()
