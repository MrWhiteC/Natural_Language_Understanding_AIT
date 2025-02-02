## Assignment 2 Deliver Details (st124903_Vorameth)


1. Task 1: Get Language Pair
    1. The dataset that is used in this assignment is scb_mt_enth_2020 (https://huggingface.co/datasets/airesearch/scb_mt_enth_2020) which published by AIResearch Thailand in the huggingface datasets. This dataset was crawled through various sources i.e. Wikipedia, SMS, Thai Websites, etc.

    ```python
    dataset = datasets.load_dataset('scb_mt_enth_2020','enth')
    ```

    2. For preparing the dataset, the library named PyThaiNLP will be used for word segmentation, tokenization, and normalization. In PyThaiNLP, the function for word_tokenize will be used for transform text into a word. The engnie is 'newmm' which will before word segmentation with dictionary-based which the function will maxmimum matching. The engnie also got constrints by Thai Character Cluster (TCC) boundaries. Moreover, the white space will be omitted which could be mapped in the English token.

    ```python
    token_transform[TRG_LANGUAGE] = pythainlp.tokenize.word_tokenize
    ```
   

2. Task 2: Experiment with Attention Mechanisms
    1. General Attention
        ```python
        energy = (encoder_outputs*hidden).sum(dim=2)
        ```

    2. Multiplicative Attention
        ```python
        energy = (encoder_outputs*self.W(hidden).repeat(1, 1, 2)).sum(dim=2)
        ```
    3. Additive Attention 
        ```python
        energy = self.v(torch.tanh(self.W(hidden) + self.U(encoder_outputs))).squeeze(2)
        ```

3. Task 3: Evaluation and Verification
    1. Metrices
        Attentions | Training Loss | Training PPL | Validation Loss | Validation PPL | Computation Time
        --- | --- | --- | --- |--- |---
        General Attention | 5.723 | 305.838 | 5.721 | 305.170 | 23 min |
        Multiplicative Attention | 5.713 | 302.825| 5.721 | 305.200 | 46 min |
        Additive Attention | 5.728 | 307.352 | 5.705 | 300.445 | 70 min |

    2. Performance Graphs
        1. General 
            ![general](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/train_loss_general.png)
        2. Multiplicative 
            ![multi](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/train_loss_multiplicative.png)
        3. Additive 
            ![add](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/train_loss_additive.png)

    3. Attention Maps
        1. General 
            ![generalmap](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/heatmap_general.png)
        2. Multiplicative 
            ![multimap](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/heatmap_multiplicative.png)
        3. Additive 
            ![addmap](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/heatmap_additive.png)

    4. Aanlysis




4. Task 4: Machine Translation - Web Application Development

    - This website build based an Attention Mechanism. The following steps show how the input text (English) will be translated (Thai).
        1. Users must input English words that would like to tranlsate into Thai words.
        2. Each word will be feed into the model to encode into a vector and then decode according to probability of targeted word in Thai.
        3. The word will provide possbile of target words based on attention score of input word and other weights. 
        4. The highest probability words will be selected and join into the sentence which will show on the interface.

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment2/images/a3_website.png)

    