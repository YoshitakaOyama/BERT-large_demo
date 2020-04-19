import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import textwrap
from pprint import pprint
import sys

# $$$ 参考にしたurl $$$
URL = "https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/"

# === 微調整された Bert-largeモデルのロード ===
# 質問応答については、SQuADベンチマーク用にすでに微調整されたバージョンのBERT-largeがある
# BERT-largeは本当に大きい。24層で埋め込みサイズは1,024で、合計で340Mのパラメーター, 全体で1.34GB
# 注：このモデルはSQuADのバージョン1でトレーニングされた
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# ======

# === トークナイザーのロード ===
# このモデルの語彙は、bert-base-uncasedの語彙とまったく同じ
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# ======

# === 解きたい質問とその回答を含んだ文章(リファレンス) ===
# QAの例は、質問とその質問への回答を含むテキストの節で構成される
# 我々は両方に対してBERTのトークナイザを実行する必要がある
question = "How many parameters does BERT-large have?"
answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."
# ======

# === 質問とリファレンスに対して、[SEP]トークンを配置し、ID付けする ===
# これらをBERTにフィードするために、実際にそれらを連結し、その間に特別な[SEP]トークンを配置
# トークナイザーのencodeで一瞬でidづけできる
# Apply the tokenizer to the input text, treating them as a text-pair.
input_ids = tokenizer.encode(question, answer_text)
print('The input has a total of {:} tokens.'.format(len(input_ids)))
# ======

# === トークナイザーの把握 ===
# トークナイザーが何をしているかを正確に確認するために、IDを付けてトークンを出力
# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)

# For each token and its id...
for token, id in zip(tokens, input_ids):
    
    # If this is the [SEP] token, add some space around it to make it stand out.
    if id == tokenizer.sep_token_id:
        print('')
    
    # Print the token string and its ID in two columns.
    print('{:<12} {:>6,}'.format(token, id))

    if id == tokenizer.sep_token_id:
        print('')

# ちなみに出力見てみると ## が埋め込まれているけど、これは繋がっている単語の時に付与される
# ======

# === A, Bの埋め込み ===
# いわゆる segment embedding と呼ばれるものですな
# We’ve concatenated the question and answer_text together, 
# but BERT still needs a way to distinguish them. 
# BERT has two special “Segment” embeddings, one for segment “A” and one for segment “B”. 
# Before the word embeddings go into the BERT layers, 
# the segment A embedding needs to be added to the question tokens, and 
# the segment B embedding needs to be added to each of the answer_text tokens.

# These additions are handled for us by the transformer library, 
# and all we need to do is specify a ‘0’ or ‘1’ for each token.
# Note: In the transformers library, huggingface likes to call these token_type_ids, 
# but I’m going with segment_ids since this seems clearer, and is consistent with the BERT paper.

# [SEP]トークンIDのインデックス
# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)

# セグメントAの数([SEP]トークン自身はAに含まれる)
# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1

# セグメントBは残りだよね
# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a

# セグメントA, Bを0, 1で表現し、それのリストを作る
# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b

# セグメントA + Bの数はトークンの数(今回は70個かな)と一致する
# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)

# ======

# === モデルを試す ===
# Run our example through the model.
# 上記で定義したモデルにinput_idsとセグメントリストを渡すと、スタートとエンドのスコア一覧のリストが返ってくる！ 超簡単
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

# scoresがどんな感じか確認
print("start scores: ")
print(f"length -> {len(start_scores)}") # tensor型なので 1
pprint(start_scores)
print("--------------")
print("end scores: ")
print(f"length -> {len(end_scores)}") # tensor型なので 1
pprint(end_scores)

# argmaxをとり、スタートとエンドのIDを取得
# Find the tokens with the highest `start` and `end` scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.
answer = ' '.join(tokens[answer_start:answer_end+1])

# print('Answer: "' + answer + '"') <-- ## 付きだけど、答え出せた！

# Start with the first token.
answer = tokens[answer_start]

# subwordトークンである##をなくして綺麗に出力
# Select the remaining answer tokens and join them with whitespace.
for i in range(answer_start + 1, answer_end + 1):
    
    # If it's a subword token, then recombine it with the previous token.
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    
    # Otherwise, add a space then the token.
    else:
        answer += ' ' + tokens[i]

print('Answer: "' + answer + '"')

# ======

# === スコアのvisuallize化(barplot) ※スタートとエンド別々に描画 ===
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
#sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (16,8)

# 上記で確認したようにtensor型なので、描画するためフラットンする
# Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
s_scores = start_scores.detach().numpy().flatten()
e_scores = end_scores.detach().numpy().flatten()
print("s_scores: ")
pprint(s_scores)
print(f"length: {len(s_scores)}") # フラットンしたので当然70になる
print("---------------------------")
print("e_scores: ")
pprint(e_scores)
print(f"length: {len(e_scores)}") # フラットンしたので当然70になる

# トークンラベルの付与
# We'll use the tokens as the x-axis labels. In order to do that, they all need
# to be unique, so we'll add the token index to the end of each one.
token_labels = []
for (i, token) in enumerate(tokens):
    token_labels.append('{:} - {:>2}'.format(token, i))

# スタートスコア
# Create a barplot showing the start word score for all of the tokens.
ax = sns.barplot(x=token_labels, y=s_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('Start Word Scores')

# plt.show()
plt.savefig('./output/start_word_score.png')

# エンドスコア
# Create a barplot showing the end word score for all of the tokens.
ax = sns.barplot(x=token_labels, y=e_scores, ci=None)

# Turn the xlabels vertical.
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
ax.grid(True)

plt.title('End Word Scores')

# plt.show()
plt.savefig('./output/end_word_score.png')
# ======

# === スコアのvisuallize化(barplot) ※スタートとエンド一緒に描画 ===
# Store the tokens and scores in a DataFrame. 
# Each token will have two rows, one for its start score and one for its end
# score. The "marker" column will differentiate them. A little wacky, I know.
scores = []
for (i, token_label) in enumerate(token_labels):

    # Add the token's start score as one row.
    scores.append({'token_label': token_label, 
                   'score': s_scores[i],
                   'marker': 'start'})
    
    # Add  the token's end score as another row.
    scores.append({'token_label': token_label, 
                   'score': e_scores[i],
                   'marker': 'end'})
    
df = pd.DataFrame(scores)

# Draw a grouped barplot to show start and end scores for each word.
# The "hue" parameter is where we tell it which datapoints belong to which
# of the two series.
g = sns.catplot(x="token_label", y="score", hue="marker", data=df,
                kind="bar", height=6, aspect=4)

# Turn the xlabels vertical.
g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

# Turn on the vertical grid to help align words to scores.
g.ax.grid(True)

plt.savefig('./output/start_end_score.png')
# ======
