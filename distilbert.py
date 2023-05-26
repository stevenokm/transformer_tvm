from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)
model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)

question = "What is a good example of a question answering dataset?"
text = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = torch.argmax(outputs.start_logits)
answer_end_index = torch.argmax(outputs.end_logits)


predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print(tokenizer.decode(predict_answer_tokens))
