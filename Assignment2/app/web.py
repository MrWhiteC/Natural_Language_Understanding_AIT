from flask import Flask, request, jsonify, render_template
import torch
from skipgram import Skipgram,get_embed,find_similar_word,SkipgramNeg,get_embed_skipgram_neg

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    model = torch.load("model_skipgram_neg", weights_only=False)
    # try:
    data = request.json
    input_word = data.get('word')
    sim_score = find_similar_word(model,input_word)
    similar_word = [i[0] for i in sim_score]

    if not input_word:
        return jsonify({"error": "Word is required"}), 400

    # Replace the following line with your model's similarity computation logic

    similarity_score = [i[1] for i in sim_score]

    return jsonify({
        "input_word": input_word,
        "similar_word": similar_word,
        "similarity_score": similarity_score
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



