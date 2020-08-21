import numpy as np
import pandas as pd
import re
import random
import time
import math
import unicodedata
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F

#User Function
from dataUtils import *
from models import EncoderRNN,LuongAttnDecoderRNN

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

################################ SET HYPER PARAMS

class HParams():
    def __init__(self):
        self.n_layers = 2
        self.hidden_size = 512
        self.fc_size = 512
        self.dropout = 0.1
        self.batch_size = 64
        self.lr = 0.0001
        self.lr_decay = 0.9999
        self.min_lr = 0.00001
        self.grad_clip = 5.
        self.cuda = True
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        self.max_length = 20
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.n_iteration = 5000
        self.print_every = 1
        self.save_every = 500
        self.decoder_learning_ratio = 5.0


hp = HParams()

data = pd.read_csv('persuader.tsv', sep='\t')
VOC = Voc(data)

loadFilename = '5000_checkpoint.tar'
#checkpoint = torch.load(loadFilename)
checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
VOC.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
embedding = nn.Embedding(VOC.num_words, hp.hidden_size)
embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(hp.hidden_size, embedding, hp.n_layers, hp.dropout)
decoder = LuongAttnDecoderRNN(hp.attn_model, embedding, hp.hidden_size, VOC.num_words, hp.n_layers, hp.dropout)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
         
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        
        decoder_hidden = encoder_hidden[:decoder.n_layers]
       
        
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
       
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        
        for _ in range(max_length):
           
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_outputs是(batch=1, vob_size)
           
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
         
            decoder_input = torch.unsqueeze(decoder_input, 0)
     
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=hp.max_length):

    indexes_batch = [indexesFromSentence(voc, sentence)]
 
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
   
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
   
    tokens, scores = searcher(input_batch, lengths, max_length)
 
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    i = 0
    while(1):
        try:
            if i ==0:
              input_sentence = normalize_string('SOD')
            # 得到用户终端的输入
            input_sentence = input('> ')
            # 是否退出
            if input_sentence == 'q' or input_sentence == 'quit': break
            # 句子归一化
            input_sentence = normalize_string(input_sentence)
            # 生成响应Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 去掉EOS后面的内容
            words = []
            for word in output_words:
                if word == 'EOS':
                    break
                elif word != 'PAD':
                    words.append(word)
            print('Bot:', ' '.join(words))

        except KeyError:
            print("Error: Encountered unknown word.")




encoder.eval()
decoder.eval()

#del input
def getoutput(msg):
    msg = normalize_string(msg)
    try:
        output_words = evaluate(encoder, decoder, searcher, VOC, msg)
        words = []
        for word in output_words:
            if word == 'EOS':
                break
            elif word != 'PAD':
                words.append(word)
        #output_words[:] = [x for x in output_words if not (x=='EOS') or x == 'PAD']
        if words[0] == 'eod':
            words = ['Byebye!']
    except KeyError:
        words = ["Sorry, I couldn't understand."]
    return ' '.join(words)

searcher = GreedySearchDecoder(encoder, decoder)
#evaluateInput(encoder, decoder, searcher, VOC)


#Creating GUI with tkinter
import tkinter

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = getoutput(msg = msg)
        #res = evaluatex(loaded_encoder, loaded_decoder, searcher, VOC, "hello")
        #res[:] = [x for x in res if not (x == 'EOS' or x == 'PAD')]
        #ChatLog.insert(END, "Bot: " + ''.join(res) + '\n\n')
        ChatLog.insert(END, "Bot: "+ res + '\n\n')
        print(res)
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

from tkinter import *
base = Tk()
base.title("Persuasive Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()