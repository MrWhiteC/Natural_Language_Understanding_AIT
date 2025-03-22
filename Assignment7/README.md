# Assignment 7 Deliver Details (st124903_Vorameth)

In this assignment, we will explore the comparison between Odd Layer and Even Layer Student Training
Models and LoRA (Low-Rank Adaptation) on a distillation task using BERT from Huggingface.

## Task 1: Hate Speech/Toxic Comment Dataset

In task 1, the datasets in this assignment is Hate Speech or Toxic Comment which could be used in training for classification problems in order to tell whether the sentences are identify as toxic or non-toxic or not. The dataset was published on Huggingface by OxAI Safety Hub which they collect the dataset from Kaggle ([Toxi Comment Classification Comment](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview) from 2017/2018). 

Dataset Detail: 
- Wikipedia Forums with labels on Toxic and Non-Toxic
- Language in English
- The dataset consists of Train, Validation, and Test. 
- Total 128 K for Train, 31 K for Validation, and 63 K for Test
- The dataset represent in id, conmment text, and label. 

```python
task_to_keys = {
    "OxAISH-AL-LLM/wiki_toxic": ("comment_text",None),
}

task_name = "OxAISH-AL-LLM/wiki_toxic"
```

## Task 2: Odd Layer vs Even Layer Training

The distillation had been done through a selecting only odd layer and even layer to be trained. In this assigment which based on code on [distilBERT.ipynb](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/blob/main/Code/04%20-%20Huggingface/Appendix%20-%20Efficient%20Network%20Training/01-distilation/distilBERT.ipynb), the BERT will be used as the teacher model which consist of 12 layers. The tokenizer will used the bert-base-uncased as well. 

```python
teacher_id = "bert-base-uncased"

model

# (0-11): 12 x BertLayer(
# (attention): BertAttention(
#     (self): BertSdpaSelfAttention(
#     (query): Linear(in_features=768, out_features=768, bias=True)
#     (key): Linear(in_features=768, out_features=768, bias=True)
#     (value): Linear(in_features=768, out_features=768, bias=True)
#     (dropout): Dropout(p=0.1, inplace=False)
# )
```
In the prepocessing, the tokenizers will be done to create input_ids, token_type_ids, and attention_mask where the other column will be delete except the label. 

For student model, the configuration will be copied from teacher model where the hidden layer will be reduced by half. 
- In even model, all the weight will be copy from the even layers.
    ```python
    student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    ```
- In odd model, all the weight will be copy from the odd layers.
    ```python
    student_encoding_layers[i].load_state_dict(teacher_encoding_layers[(2*i)-1].state_dict())
    ```

This result in half of the number of the models for student models.
Teacher parameters : 109483778
Student parameters Even : 66956546
Student parameters Odd : 66956546

## Task 3: LoRA (Low-Rank Adaptation)

LoRA will be applied into teacher model which originally consist of 12 layers which the LoRA will performs trainable rank decomposition into each layer. 

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(teacher_model, peft_config)
```

## Task 4: Evaluation and Analysis

- Evaluation

    Model Type | Training Loss | Test Set Performance
    --- | --- | --- 
    Odd Layer | Train loss 0.1771, Loss_cls: 0.3186, Loss_div: 0.1497, Loss_cos: 0.0630 |  Accuracy =  0.959|
    Even Layer | Train loss 0.1775, Loss_cls: 0.3201, Loss_div: 0.1492, Loss_cos: 0.0632|  Accuracy =  0.9570000000000001|
    LoRA | Train loss 0.2507, Loss_cls: 0.7522, Loss_div: 0.0000, Loss_cos: -0.0000|  Accuracy =   0.11640000000000002|

    Based on the model, Odd model return with the best performance in term of loss and accuracy. Even model follow with second model, while LoRA model perform the worst compared to the other model. This could be becuase the model was trained with manual training technique like other model not trainer library. 

- Anlysis

    Challenge : The main challenge to implment the distillation and LoRA is the implementation. While the LoRA proivdes a build in function in the PEFT library already, the distillation allow users to remove or distill the layer manually which giving a power to control the model more flexible compare the LoRA. Moreover, during the training and configuration, there are many multiple errors with LoRA technique which interupt the LoRA configuration and training process. In configuration, the LoRA faced with CUDA errors on the puffer where it cannot locate the CUDA drvier. For training, LoRA with trainer function always keep missing the size in batch. This could be becuase using the default value. 

    Improvement : To improve from afromentioned challenge, the even and odd model will be trained in puffer, while LoRA model will be tested on the local machine allowing full manipulation on the driver errors. After that, some of the LoRA default values had been changed to comply with classification problem which the model will be trained through teacher training, tokenizer will be used with teacher model, and batch or support function will be used same as teacher model. However, these are just work around for this assignment. In order to improve the model. The batch or support function need to be input in the trainer library. In addition, the current samples are quite small due to resource limitation. Adding more samples could improve the model accuracy.


## Task 5: Web Application

The website is a classification problems which will classify the comment or text which might be toxic or non-toxic based on the BERT model.

Input Text: Question

Result : Classification on Toxic or Non-Toxic


Example 1

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment7/images/website1.png)

Example 2

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment7/images/website2.png)

Example 3

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment7/images/website3.png)

    