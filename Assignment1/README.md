## Assignment 1 Deliver Details


1. Task 1
    1. Training with a real-world corpus 
        The real-world corups was used in this assigment is Brown Copus (only news category) which downloaded from NTLK datasets.
        ![real_corpus](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment1/images/a1_brown_corpus.png)
    2. Windows Size modification
        The windows size was add in the jupyter notebook in random_batch function as for allowing the implementer adjust the word that pair up with center word for vary the embedding.
        ![window_size](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment1/images/a1_window_size.png)
2. Task 2 

The table are provided as the comparsion of the model in term of Training Loss, Training Time, Syntactic, and Semantic Accuracy. Due to the limit word in the copus, the accurancy is alway return zero result.

Model | Window Size | Training Loss | Training Time | Syntactic Accuracy | Semantic Accuracy
--- | --- | --- | --- |--- |--- 
Skipgram | 4 | 10.606616 | 6 min 44 sec | 0.19762 | 0.0641 |
Skipgram (NEG) | 4 | 3.371357 | 5 min 35 sec | 0 | 0 |
GloVe          | 4 | 4.851114 | 1 min 37 sec | 0 | 0 |
GloVe (Gensim) | - | -  | -  | 0 | 2.30769 |

The most correlate between human judgement and the model is the Skipgram (Negative Sampling). This model will be used in the website.

Model |   Skipgram |   Skipgram (NEG) |   GloVe |   GloVe (Gensim) |   Y true |
--- | --- | --- | --- |--- |--- 
MSE |  0.0214 |     0.1019 |           0.1018 |           -0.577 |        1 |

3. Task 3

The website is developed for allowing user to search for 10 similar words through the dot product in the corpus and input word. 

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment1/images/a1_website.png)