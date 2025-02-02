
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:
    raise ModuleNotFoundError("PyTorch is not installed. Please install it using 'pip install torch'.")

import sentencepiece as spm
from datasets import load_dataset
from flask import Flask, request, jsonify
import os

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')

# Ensure dataset is downloaded locally
dataset_path = "./data"
os.makedirs(dataset_path, exist_ok=True)
dataset = load_dataset("airesearch/scb_mt_enth_2020", "enth", cache_dir=dataset_path)

# Define Seq2Seq Model with Attention
class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(AdditiveAttention, self).__init__()
        self.W1 = nn.Linear(encoder_dim, attention_dim)
        self.W2 = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        encoder_transformed = self.W1(encoder_outputs)
        decoder_transformed = self.W2(decoder_hidden).unsqueeze(1)
        scores = self.v(torch.tanh(encoder_transformed + decoder_transformed)).squeeze(2)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, attention):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = attention

    def forward(self, decoder_input, hidden, encoder_outputs):
        embedded = self.embedding(decoder_input).unsqueeze(1)
        context_vector, attention_weights = self.attention(hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context_vector.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        output = self.fc(output.squeeze(1))
        return output, hidden.squeeze(0), attention_weights

# Load trained model (dummy model used here, replace with actual trained weights)
encoder_dim, decoder_dim, output_dim, embed_dim = 512, 512, 1000, 300
attention_mechanism = AdditiveAttention(encoder_dim, decoder_dim, 256)
decoder = DecoderWithAttention(output_dim, embed_dim, decoder_dim, attention_mechanism)

def translate_text(text):
    tokens = sp.encode(text, out_type=int)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    encoder_outputs = torch.randn(1, len(tokens), encoder_dim)
    hidden = torch.randn(1, decoder_dim)
    output, _, _ = decoder(tokens_tensor[:, 0], hidden, encoder_outputs)
    translated_tokens = output.argmax(1).tolist()
    translated_text = sp.decode(translated_tokens)
    return translated_text

# Flask API
app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    input_text = data.get('text', '')
    translation = translate_text(input_text)
    return jsonify({'translation': translation})

if __name__ == '__main__':
    app.run(debug=True)
