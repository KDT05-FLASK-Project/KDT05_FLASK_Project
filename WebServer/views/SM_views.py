from flask import Blueprint, render_template, request, url_for, redirect
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from nltk.tokenize import word_tokenize
from WebServer import db
from WebServer.models import SM_DB

bp = Blueprint('SM', __name__, template_folder = 'templates',
                    url_prefix="/sm_db")

# Define the translation dataset
class TranslationDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.src = self.data.iloc[:, 0]
        self.trg = self.data.iloc[:, 1]
        self.src_tokenizer = word_tokenize
        self.trg_tokenizer = word_tokenize
        self.word2idx = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.idx2word = {}  # Initialize the idx2word dictionary
        self.build_vocab()

    def build_vocab(self):
        for index, row in self.data.iterrows():
            src_words = self.src_tokenizer(row[0].lower()) + ["<sos>", "<eos>"]
            trg_words = self.trg_tokenizer(row[1].lower()) + ["<sos>", "<eos>"]
            for word in src_words + trg_words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word[self.word2idx[word]] = word  # Add to idx2word dictionary


    def tokenize(self, text):
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in word_tokenize(text.lower()) + ["<eos>"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = torch.tensor([self.word2idx["<sos>"]] + self.tokenize(self.src.iloc[idx]), dtype=torch.long)
        trg = torch.tensor([self.word2idx["<sos>"]] + self.tokenize(self.trg.iloc[idx]), dtype=torch.long)
        return src, trg


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_output):
        # Add batch dimension to input if necessary (making sure it's always 3D)
        if input.dim() == 1:
            input = input.unsqueeze(0)  # Add batch dimension if it's missing
        embedded = self.embedding(input)

        # Adjust hidden state dimensions if necessary
        if hidden[0].dim() == 2:
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))  # Ensure hidden is 3D by adding batch dimension

        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output.squeeze(0))  # Remove batch dimension for linear layer if needed
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_output, encoder_hidden = self.encoder(src)
        decoder_output, decoder_hidden = self.decoder(trg, encoder_hidden, encoder_output)
        return decoder_output


hidden_size = 256
encoder = Encoder(45746, hidden_size)
decoder = Decoder(45746, hidden_size)
model = Seq2Seq(encoder, decoder)

#
# state_dict = torch.load('이시명/best_model_final{}.pth', map_location=torch.device('cpu'))
#
# model.load_state_dict(state_dict)

@bp.route('/')
def index():
    return render_template('SM/index.html')



# Prepare a sentence for translation
def prepare_sentence(sentence, dataset, device):
    tokens = dataset.tokenize(sentence)
    numerical = torch.tensor([tokens], dtype=torch.long).to(device)
    return numerical

# Translate the sentence
def translate(model, src_tensor, dataset, device):
    model.eval()
    src_tensor = src_tensor.to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [dataset.word2idx['<sos>']]  # Start token

    for _ in range(100):  # Maximum length of the translated sentence
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == dataset.word2idx['<eos>']:  # End token
            break

    translated_sentence = ' '.join(dataset.idx2word.get(idx, '<unk>') for idx in trg_indexes[1:-1])  # Skip <sos> and exclude <eos>
    return translated_sentence




data10k = pd.read_csv('이시명/data10k.csv', header=None)
data_subset = data10k
dataset = TranslationDataset(data_subset)

# Example sentence



@bp.route('/detail/<int:comment_id>', methods=['GET','POST'])
def detail(comment_id):
    comment = SM_DB.query.get(comment_id)
    return render_template("SM/comment_processing.html", comment=comment)

@bp.route('/delete/<int:comment_id>', methods=['GET','POST'])
def delete_comment(comment_id):
    comment = SM_DB.query.get(comment_id)
    db.session.delete(comment)
    db.session.commit()
    table_list = SM_DB.query.order_by(SM_DB.create_date.desc())
    return redirect(url_for('SM.result'))

@bp.route('/result', methods=['POST'])
def result():
    textwrap = request.form.to_dict()
    text = textwrap.get('input')
    text2 = textwrap.get('numbers')

    state_dict = torch.load(f'이시명/best_model_final{text2}.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    korean_sentence = text
    input_tensor = prepare_sentence(korean_sentence, dataset, 'cpu')

    # Output translation
    translation = translate(model, input_tensor, dataset, 'cpu')
    q = SM_DB(input=text, output=translation, create_date=datetime.now())
    db.session.add(q)
    db.session.commit()
    table_list = SM_DB.query.order_by(SM_DB.create_date.desc())
    return render_template("SM/result.html", korean_sentence=korean_sentence, translation=translation, table_list=table_list )