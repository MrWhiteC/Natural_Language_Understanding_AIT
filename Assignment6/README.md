# Assignment 6 Deliver Details (st124903_Vorameth)

In this assignment, apply RAG (Retrieval-Augmented Generation) techniques in Langchain framework to
augment your chatbot that specializes in answering questions related to yourself, your documents, resume,
and any other relevant information.

## Task 1: Source Discovery

In task 1, all the suitable relevant sources of myself will be collected for loading into the model through RAG techniques. This could create a chatbot that contains a knowledge base of myself or, in other words, the cloned chatbot. The cloned chatbot will provide a answer based on the personal information tailoring through a RAG techniques and prompt format.

Here are all the relevant sources of myself:

- Researchgate Website (https://www.researchgate.net/profile/Vorameth-Reantongcome)
- Linkein Website (https://www.linkedin.com/in/vorameth-reantongcome/)
- PDF of CV 

In addition, a template of the prompt will be customized for allowing the chatbot cloned to be myself. So, the context will be a predefined text that allow the model know who it will pretend to be. Then, the question will be asked from a human. This will allow the model to think of myself only which focusing on answer the question based on the document from mutliple sources.

```python
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            " I'm your clone myself VoramethBot. You can ask anything according to yourself. "
            "Sometimes, it is ok to feel loss, but remember to speak to  who might knock in a sane mind again. "
            "You can ask whether who you are ? what are your studies ? what are your interest ? what expertis areas ?  "
            "No matter you feel uncomfortble, please reach out to me, ok ? {context}.",
        ),
        ("human", "{question}"),
    ]
)
```

After the design of the template, some models will be explored for a choosing best chatbot's response. Here are the models that have been explored.  
- fastchat-t5-3b-v1.0
- llama-3.1-8b-instant
- llama3-70b-8192 (Selected)

The llama3-70b-8192 was choosen to be used in this assignment because it could return the most relevant information according RAG. 


## Task 2: Analysis and Problem Solving

In this assignment there 3 generator models and 2 retreiver models that had been utilized to create a cloned chatbot of myself.

- Retreiver
    1. FAISS
    2. LancDB
    3. Chroma

- Generator Models
    1. FastChat-T5
    2. GROQ - llama-3.1-8b-instant
    3. GROQ - llama3-70b-8192

## Task 3: Chatbot Development - Web Application Development

The website will be built for allowing user to fill the question which the system will anwser according to the trained model on DPO with training dataset on task 2 and task 1. 

Input Text: Question

Result : Chatbot Answer with Sources

All Question  and Answer Link :

![Question and Answer](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment6/question_answer.json)

Example 1

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment6/images/website1.png)

Example 2

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment6/images/website2.png)

Example 3

![website](https://github.com/MrWhiteC/Natural_Language_Understanding_AIT/blob/main/Assignment6/images/website3.png)

    