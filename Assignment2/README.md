## Assignment 2 Deliver Details


1. Task 1: Data Acquisition 
    1. The dataset that is used in the books that have a story about general fairly tales in the past. The ebook was provided by The Project Gutenberg which allowed anyone to download the publication with .txt file. The file could be used to train in this language model using Long Short-Term Memory (LSTM). 

    List of books for training: 
    - https://www.gutenberg.org/cache/epub/503/pg503.txt
    - https://www.gutenberg.org/cache/epub/11027/pg11027.txt
    - https://www.gutenberg.org/cache/epub/52719/pg52719.txt 
    - https://www.gutenberg.org/cache/epub/19734/pg19734.txt
    - https://www.gutenberg.org/cache/epub/31431/pg31431.txt
    - https://www.gutenberg.org/cache/epub/33511/pg33511.txt
    - https://www.gutenberg.org/ebooks/24778

2. Task 2: Model Training
    1. Prepocessing Steps
        1. Load Data - The very first step on creating any model is loading the data. In this language model. I use library of Huggingface for text loader function which will feed the text in each row into the Huggingface dict of train, test and validate.

        2. Tokenization - After the text had been loaded, the tokenization with basic english will be performed to toknize each row of sentence to be just a simple english word. These tokenized text will be next add into the vocaburary word for next step of preprocessing. 

        3. Numericalizaing - After tokenization, the vocaburary data will be generated according to the tokenization where the frequent number of the appeared word will be insert in the vocabuarary. The vocaburary will be used in the data batch and index on prediction.

    2. Model Architecture and Training Process
        1. Prepare data - The data from the train data will be divide according to the batch size. This allows the tokenized data that will be put into the sequnce number in each batch size. 
        2. Model Architecture - The model consist of 3 part consisting of embedding, lstm, linear.
            1. Embedding - In this layer, the one word of each squence of text in each batch will be feed into the embedding network for computing the vecotr of the text. 
            2. LSTM - Then, the LSTM layer will receive the vector from embedding layer and will compute the output word accoring to the equation through forget gate, input gate, output gate and result through current and previous cells to calculate the output cell. The result of each word will be feed as the initial controlling factor of forgeting or remembering for the next word. 
            3. Linear - Finally, after computation of sequnce of word, the linear layer will be compute to most probable of word through softmax function which will used as the predicted words. 

        3. Training Process
            1. The training process will run through sequnce of the word input and validate with the next sequnce of the word. In each batch, the sequnce of word will be feed into the model. Then, the prediction of the word will be check with the target in order to calculate the loss.

3. Task 3: Text Generation - Web Application Development

    