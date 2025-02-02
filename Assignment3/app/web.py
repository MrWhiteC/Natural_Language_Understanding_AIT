from flask import Flask, request, jsonify, render_template
import torch,datasets
from sq2sq import Seq2SeqPackedAttention,Attention,Encoder,Decoder
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchdata.datapipes.iter import IterableWrapper, ShardingFilter
import pythainlp
import sys

SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'th'
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']

emb_dim     = 256  
hid_dim     = 512  
dropout     = 0.5
SRC_PAD_IDX = PAD_IDX

attn = Attention(hid_dim)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

text_transform = {}
token_transform = {}
vocab_transform = {}

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

dataset = datasets.load_dataset('scb_mt_enth_2020','enth')
test = dataset["train"]['translation'][: 1000]
listtuple = list()
for i in test:
    listtuple.append((i['en'],i['th']))
datapipe = IterableWrapper(listtuple)
train =  datapipe.sharding_filter()

train_size = len(list(iter(train)))

train, val, test = train.random_split(total_length=train_size, weights = {"train": 0.7, "val": 0.2, "test": 0.1}, seed=999)

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TRG_LANGUAGE] = pythainlp.tokenize.word_tokenize

def yield_tokens(data, language):
    language_index = {SRC_LANGUAGE: 0, TRG_LANGUAGE: 1}
    if language == 'en':
        for data_sample in data:
            yield token_transform[language](data_sample[language_index[language]]) #either first or second index
    if language == 'th':
        for data_sample in data:
            yield token_transform[language](data_sample[language_index[language]],engine="newmm")

for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    # Create torchtext's Vocab object 
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train, ln), min_freq=2, specials=special_symbols, special_first=True)                                           

for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor
    
input_dim   = len(vocab_transform[SRC_LANGUAGE])
output_dim  = len(vocab_transform[TRG_LANGUAGE])
enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
dec  = Decoder(output_dim, emb_dim,  hid_dim, dropout, attn)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():

    # try:
    data = request.json
    input_word = data.get('word')

    prompt = input_word
    max_seq_len = 30
    seed = 0
    
    model = Seq2SeqPackedAttention(enc, dec, SRC_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load('models/Seq2SeqPackedAttention.pt'))



    src_text = text_transform[SRC_LANGUAGE](input_word).to(device)
    
    trg_text = text_transform[TRG_LANGUAGE](input_word).to(device)

    src_text = src_text.reshape(-1, 1)  
    trg_text = trg_text.reshape(-1, 1)

    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)

    model.eval()

    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0) #turn off teacher forcing
    
    output = output.squeeze(1)
    output = output[1:]
    output_max = output.argmax(1)

    mapping = vocab_transform[TRG_LANGUAGE].get_itos()

    similar_word = ''

    for token in output_max:
        similar_word = similar_word + mapping[token.item()]


    if not input_word:
        return jsonify({"error": "Word is required"}), 400

    # Replace the following line with your model's similarity computation logic
    # for token in output_max:
    #  print(mapping[token.item()],file=sys.stderr)


    return jsonify({
        "input_word": input_word,
        "similar_word": similar_word,
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



