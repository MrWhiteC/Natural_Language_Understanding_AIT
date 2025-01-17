# Natural_Language_Understanding_AIT
Assignment Submission for NLU Class 2025


Model | Window | Training Loss | Training time | Syntactic Accuracy | Semantic accuracy
--- | --- | --- | --- |--- |--- 
Skipgram | 4 | 10.606616 | 6 min 44 sec | 0.19762 | 0.0641 |
Skipgram (NEG) | 4 | 3.371357 | 5 min 35 sec | 0 | 0 |
GloVe          | 4 | 4.851114 | 1 min 37 sec | 0 | 0 |
GloVe (Gensim) | 4 | -  | -  | 0 | 2.30769 |


Model |   Skipgram |   Skipgram (NEG) |   GloVe |   GloVe (Gensim) |   Y true |
--- | --- | --- | --- |--- |--- 
MSE |  0.0214 |     0.1019 |           0.1018 |           -0.577 |        1 |