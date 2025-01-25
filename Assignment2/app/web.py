from flask import Flask, request, jsonify, render_template
import torch
from lstm import LSTMLanguageModel,generate,get_tokenizer,get_vocab
import datasets

dataset = dataset = datasets.load_dataset( "text", data_files={"train": "fairytales_train.txt","test":"fairytales_test.txt","validation":"fairytales_validate.txt"})
emb_dim = 1024                
hid_dim = 1024                
num_layers = 2                
dropout_rate = 0.65              
lr = 1e-3            
tokenizer = get_tokenizer()
vocab = get_vocab(dataset)
vocab_size = len(vocab)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
    model.load_state_dict(torch.load('best-val-lstm_lm.pt',  map_location=device))
    try:
        data = request.json
        input_word = data.get('word')

        prompt = input_word
        max_seq_len = 30
        seed = 0
        
        generation = generate(prompt, max_seq_len, 0.5, model, tokenizer, vocab, device, seed)


        similar_word = ' '.join(generation)

        if not input_word:
            return jsonify({"error": "Word is required"}), 400

        # Replace the following line with your model's similarity computation logic

        similarity_score = 1

        return jsonify({
            "input_word": input_word,
            "similar_word": similar_word,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



