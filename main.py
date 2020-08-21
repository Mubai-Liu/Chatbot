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

################################# READ DATA
data = pd.read_csv('persuader.tsv', sep='\t')

data['input'] = data['input'].apply(normalize_string)
data['target'] = data['target'].apply(normalize_string)

NUM_LINES = len(data)
print(NUM_LINES)

VOC = Voc(data)
pairs = []
for input,target in zip(data['input'],data['target']):
    VOC.addSentence(input)
    VOC.addSentence(target)
    if len(input.split(' '))<hp.max_length and len(target.split(' '))<hp.max_length:
        pairs.append([input,target])

print(len(pairs))
print(random.choice(pairs))
print(VOC.num_words)


def maskNLLLoss(inp, target, mask):

    nTotal = mask.sum()
    
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=hp.max_length):

    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0


    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Decoder的初始输入是SOS，我们需要构造(1, batch)的输入，表示第一个时刻batch个输入。
    decoder_input = torch.LongTensor([[SOS_token for _ in range(hp.batch_size)]])
    decoder_input = decoder_input.to(device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

 
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

 
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_input = target_variable[t].view(1, -1)
          
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
           
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
          
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    
    loss.backward()
    

   
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

   
    encoder_optimizer.step()
    decoder_optimizer.step()
    encoder_scheduler.step()
    decoder_scheduler.step()
    return sum(print_losses) / n_totals

def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 
              embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, 
              print_every, save_every, clip, corpus_name, loadFilename):

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]


    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1


    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        
        input_variable, lengths, target_variable, mask, max_target_len = training_batch


        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss


        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
			.format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # 保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'
		.format(encoder_n_layers, decoder_n_layers, hp.hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))



print('Building encoder and decoder ...')
# word embedding
embedding = nn.Embedding(VOC.num_words, hp.hidden_size)

encoder = EncoderRNN(hp.hidden_size, embedding, hp.n_layers, hp.dropout)
decoder = LuongAttnDecoderRNN(hp.attn_model, embedding, hp.hidden_size, VOC.num_words, 
			hp.n_layers, hp.dropout)

encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')



encoder.train()
decoder.train()


print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=hp.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=hp.lr * hp.decoder_learning_ratio)
encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, 5)
decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, 5)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


print("Starting Training!")
trainIters(hp.model_name, VOC, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, hp.n_layers, hp.n_layers, 'savedModels/checkpoint', hp.n_iteration, hp.batch_size,
           hp.print_every, hp.save_every, hp.clip, 'persuade', loadFilename)
