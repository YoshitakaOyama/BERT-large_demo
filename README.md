[Reference](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/)

# Execution procedure (Question Answering on BERT-large)
1. Install required python package.
2. If you want to try out the model and tokenizer specifications, or visualize the start and end scores for every word, run
```
python demo.py
```
3. To do QA,　Pass the question string and reference string to the `answer_question function` in `predict.py` and execute it. The answer comes back. The model and tokenizer used here are automatically downloaded at the first startup ([Click here for model details](https://github.com/huggingface/transformers)).

ex.)
```
import textwrap
from bert.predict import answer_question


wrapper = textwrap.TextWrapper(width=80)

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
```
If you want to change the model or tokenizer, refer to [here](https://huggingface.co/transformers/pretrained_models.html) and pass each name as an argument and execute it.
