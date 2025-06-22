from rouge_score import rouge_scorer


def rouge(answer, expected_answer):
    '''
    :param answer: the answer produced by pipeline
    :param expected_answer: the expected answer
    :return: ROUGE-1, ROUGE-L
    '''

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(expected_answer, answer)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def bertScore(answer, expected_answer, bert_scorer):
    '''
    :param answer: the answer produced by pipeline
    :param expected_answer: the expected answer
    :param bert_scorer: BertScore-P, BertScore-R, BertScore-F1
    :return:
    '''
    P, R, F1 = bert_scorer.score([answer], [expected_answer])
    return P[0].item(), R[0].item(), F1[0].item()
