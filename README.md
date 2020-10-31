# 202005-28 Machine with Personality-Persuasive Chatbot

Check our Youtube video for more details our project.
[Link](https://youtu.be/da8ayvIP2u0)

# Introduction
Persuading someone to do something is a task that most of our lifetime will experience. Using human resources to do such a task is time consumable and inefficient. However, building and implementing a persuasive chatbot can negate such a downside of using human resources. Our project is based on this intuition and focuses on persuade the user to make a donation to children, and we believe by adding a personality aspect of the target will also increase the chance of success. Hence, by using human samples of persuading different personality people datasets, we developed a python scripted chatbot that will interact with the user and do the persuading job of donation.

This is a project that mainly focus on persuading people to make donation to children. Our model is mostly based on Seq2Seq model and we use tokenize method to help the model recognize each sentence and conversation. The above code including training part and testing part of our project. Please check the instruction below to make sure it works fine on your local machine.


# Instruction
Training
```
python main.py
 ```
 Evaluation
 ```
 python evaluation.py
 ```
persuader.tsv - Data after little preprocess

models.py - Encoder-Decoder model

dataUtils.py - All data preprocess

# Data
[Link](https://gitlab.com/ucdavisnlp/persuasionforgood)

# Detail Description
