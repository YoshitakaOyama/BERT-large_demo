from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import textwrap
import torch


def answer_question(question, answer_text, model_name=None, tokenizer_name=None):
    """
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer.

    Parameters
    ----------
    question : str
    answer_text : str
    model : str
    tokenizer : str

    Return
    -------
    answer : str
    """
    # ======== Model & Tokenizer (default: bert-large finetuned squad ver.1)========
    if model_name is None:
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    if tokenizer_name is None:
        tokenizer_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # ======== Tokenize ========
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    # print(f"Query has {len(input_ids):,} tokens.\n")

    # ======== Set Segment IDs ========
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    start_scores, end_scores = model(
        torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids])
        )

    # ======== Reconstruct Answer ========
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]
    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return answer


if __name__ == '__main__':
    wrapper = textwrap.TextWrapper(width=80)

    # example1
    bert_abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."
    print(f"bert_abstract(reference): \n{wrapper.fill(bert_abstract)}\n")

    questions = [
        "What does the 'B' in BERT stand for?",
        "What are some example applications of BERT?",
        "What are some example applications of BEßRT?"
    ]
    for question in questions:
        print(f"question: {question}")
        answer = answer_question(question, bert_abstract)
        print(f"answer: {answer}\n")

    # example2
    bert_abstract = "Private schools, also known as independent schools, non-governmental, or nonstate schools, are not administered by local, state or national governments; thus, they retain the right to select their students and are funded in whole or in part by charging their students tuition, rather than relying on mandatory taxation through public (government) funding; at some private schools students may be able to get a scholarship, which makes the cost cheaper, depending on a talent the student may have (e.g. sport scholarship, art scholarship, academic scholarship), financial need, or tax credit scholarships that might be available."
    print(f"bert_abstract(reference): \n{wrapper.fill(bert_abstract)}\n")

    questions = [
        "Along with non-governmental and nonstate schools, what is another name for private schools?",
        "Along with sport and art, what is a type of talent scholarship?",
        "Rather than taxation, what are private schools largely funded by?"
    ]
    for question in questions:
        print(f"question: {question}")
        answer = answer_question(question, bert_abstract)
        print(f"answer: {answer}\n")
