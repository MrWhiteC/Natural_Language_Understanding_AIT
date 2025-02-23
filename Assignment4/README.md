# Assignment 4 Deliver Details (st124903_Vorameth)


## Task 1:
1. 

```python
dataset = datasets.load_dataset('scb_mt_enth_2020','enth')
```

2. 

```python
token_transform[TRG_LANGUAGE] = pythainlp.tokenize.word_tokenize
```
   

## Task 2: 
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

## Task 3: 
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

4. Analysis
    According to the aformentioned metrics, performance graph, and attention maps. It seems like additive attention is the most efficient and accurate due to the overall training loss and perplexity in training and validation during training process. This is becuase in additive attention allow more path to be learning through weight whichincrease the possible answer for translation. However, in the heatmap, the result seem to show that additive fail to translate the word from the input words, but eventually I chose the additive because additive result more various words compared to other models.This could happen becuase the complication of Thai sentence which require more Thai tokenization or normalization process that will corretly mapped with English words directly. 


## Task 4: Machine Translation - Web Application Development

- This website build based an Attention Mechanism. The following steps show how the input text (English) will be translated (Thai).
    1. Users must input English words that would like to tranlsate into Thai words.
    2. Each word will be feed into the model to encode into a vector and then decode according to probability of targeted word in Thai.
    3. The word will provide possbile of target words based on attention score of input word and other weights. 
    4. The highest probability words will be selected and join into the sentence which will show on the interface.

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment3/images/a3_website.png)

    