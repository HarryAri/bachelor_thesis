from config.chat import chat

def load_plaintext(path):
    '''
    :param path: the path to the lexical database
    :return: separate each line into different "documents"
    '''
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]

def run_pt(source, question, model, model_llm, top_k=5):
    '''
    :param source: it is the path to the lexical database
    :param question: input query from the user
    :param model: model containing the splitted documents
    :param model_llm: the LLM used to generate the response
    :param top_k: number of top ranked documents fed to LLM
    :return: NPC's response to the user query
    '''
    tokenized_query = question.lower().split()
    retrieved_info = model.get_top_n(tokenized_query, source, n=top_k)
    return chat(model_llm, question, retrieved_info)
