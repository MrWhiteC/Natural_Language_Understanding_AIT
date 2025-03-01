# Assignment 5 Deliver Details (st124903_Vorameth)

This assignment focuses on using Hugging Face models to optimize human preference, specifically lever-
aging the Direct Preference Optimization (DPO) trainer. You will work with preference datasets, train a
model, and push it to the Hugging Face model hub. Additionally, you will build a simple web application
to demonstrate the trained model.

## Task 1: Finding a Suitable Dataset

In task 1, the suitable dataset will be prepared for training a pre-trained model. For this project, [XueyingJia/hh-rlhf-train-helpful-subset](https://huggingface.co/datasets/ProlificAI/social-reasoning-rlhf) will be used. This dataset was created based on helpful and harmless context. Specifically, in this dataset, the only helpful context from human prompt will be used for training the model. 

- Train : 42.4 K Rows
- Created on HuggingFace by XueyingJia
- The pre-process step is "Human: " search term will be used to fileter the prompt from human. 

```python
dataset = load_dataset('XueyingJia/hh-rlhf-train-helpful-subset', split=split)
```

In pre-processing step, the pre-trained tokenizer will be used (According to GPT2): 
- 
- 
- 



## Task 2: Training a Model with DPOTrainer

The model will be build based on the pre-trained model **('GPT2')** with the prepared dataset through **Prof. Chaklam Silpasuwanchai's** ([Code](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/blob/main/Code/07%20-%20Human%20Preferences/huggingface/04-DPO.ipynb)). 


To evluate the model, they hyperparameter will be tunned in order to minimize the loss according model monitoring in **wandb.ai**. There are 3 hyparameter that will be adjust consisting of learning rate, gradient accumulation steps, and beta. Here are the metrice table for each parameters. 

1. Learning Rate
    Parameter | Loss
    --- | --- 
    1e-1 | Loss =  0.8938|
    1e-3 | Loss = 154.5403|
    1e-6 | Loss = 24.8323|


2. Gradient Accumulation Steps
    Parameter | Loss
    --- | --- 
    1 | Loss =  0.8938|
    3 | Loss = 314.9733|
    5 | Loss = 454.8409|

3. Beta
    Parameter | Loss
    --- | --- 
    0.1 | Loss =  0.8938|
    0.3 | Loss = 242.0396|
    0.5 | Loss = 365.9005|


It can be seen that with learning rate of 1e-1, gradient accumulation steps with 1, and beta with 0.1 minimize the loss in this value. 


![loss](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment5/images/loss.png)


## Task 3: Pushing the Model to Hugging Face Hub

After the training process, the model will be used into hugginface platform for allowing for publicly used of the model. [dpo_model_rlhf](https://huggingface.co/mrwhitec/dpo_model_rlhf)


```python
model.push_to_hub("dpo_model_rlhf")
```


## Task 4: Web Application Development

The website will be built for allowing user to fill the question which the system will anwser according to the trained model on DPO with training dataset on task 2 and task 1. 

Input Text: Question

Result : Generated Text from Question

Example 1

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment5/images/website1.png)

Example 2

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment5/images/website2.png)

Example 3

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment5/images/website3.png)

    