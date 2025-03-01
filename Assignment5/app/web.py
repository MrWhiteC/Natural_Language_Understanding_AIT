from flask import Flask, request, jsonify, render_template
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():

    # try:
    data = request.json
    sentence = data.get('word')
    model = AutoModelForCausalLM.from_pretrained("saved_model")
    tokenizer = AutoTokenizer.from_pretrained("saved_model")
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


    if not sentence:
        return jsonify({"error": "Word is required"}), 400


    return jsonify({
        "input_worda": sentence,
        "similar_word": generated_text
    })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)



