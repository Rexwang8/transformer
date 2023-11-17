import pickle
import time
from matplotlib import pyplot as plt, ticker
import torch
import numpy as np
import pandas
import math
import os
from transformers import GPT2Tokenizer
import json
import random
from deep_translator import GoogleTranslator
from unidecode import unidecode
from nltk.translate.bleu_score import corpus_bleu
def preinit():
    #check if cuda is available
    device = None
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device('cuda')
    else:
        print("CUDA is not available")
        exit(1)
        device = torch.device('cpu')
        
    
    #set random seed
    RANDOMSEED = random.randint(0, 100000)
    torch.random.seed()

    #torch.manual_seed(0)
    #numpy.random.seed(0)
    #torch.cuda.manual_seed(0)
    #torch.cuda.manual_seed_all(0)
    #torch.backends.cudnn.deterministic = True
    
    #set torch options
    torch.set_printoptions(precision=10)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.set_grad_enabled(True)
    torch.set_default_device(device)
    
    return device


def loadDatasets(train=True, startToken="<|startoftext|>", endToken="<|endoftext|>"):
    #we want to load the english and german datasets
    if train:
        engpath = os.path.join(os.getcwd(), "flores200_dataset", "dev", "eng_Latn.dev")
        germanPath = os.path.join(os.getcwd(), "flores200_dataset", "dev", "deu_Latn.dev")
    else:
        engpath = os.path.join(os.getcwd(), "flores200_dataset", "devtest", "eng_Latn.dev")
        germanPath = os.path.join(os.getcwd(), "flores200_dataset", "devtest", "deu_Latn.dev")
    
    #load the datasets
    eng = pandas.read_csv(engpath, sep='\t', header=None)
    german = []
    with open(germanPath, 'r', encoding='latin1') as f:
        for line in f:
            #clean up broken characters
            line = line.replace('�\x80\x9e', '"')
            line = line.replace('�\x80\x9c', '"')
            line = line.replace('�\x80\x93', '-')
            line = line.replace('�\x9f', 'ß')
            line = line.replace('�\x9f', 'ß')
            line = line.replace('�\xad', 'í')
            german.append(line.strip())
    
    #create eng-german pairs
    pairs = []
    for i in range(len(eng)):
        pairs.append([eng.iloc[i][0], german[i]])
        
    #for both, add start and end tokens
    #for i in range(len(pairs)):
        
        

    
    #return the pairs
    return pairs

def countVocab(dataset, tokenizer):
    vocab_en = {}
    vocab_de = {}
    for pair in dataset:
        word_list_en = pair[0].split()
        word_list_de = pair[1].split()
        for word in word_list_en:
            if word in vocab_en:
                vocab_en[word] += 1
            else:
                vocab_en[word] = 1
        for word in word_list_de:
            if word in vocab_de:
                vocab_de[word] += 1
            else:
                vocab_de[word] = 1
                
    tokenizedVocab_en = {}
    tokenizedVocab_de = {}
    for pair in dataset:
        tokens_en = tokenizer.encode(pair[0])
        tokens_de = tokenizer.encode(pair[1])
        for token in tokens_en:
            if token in tokenizedVocab_en:
                tokenizedVocab_en[token] += 1
            else:
                tokenizedVocab_en[token] = 1
        for token in tokens_de:
            if token in tokenizedVocab_de:
                tokenizedVocab_de[token] += 1
            else:
                tokenizedVocab_de[token] = 1
                
        #pad the vocab to be the right size
        max_en = max(vocab_en.values())
        max_de = max(vocab_de.values())
        max_token_en = max(tokenizedVocab_en.values())
        max_token_de = max(tokenizedVocab_de.values())
        minsize= 50257
        pad_token = 2
        
        for i in range(1, max(minsize, max_en)+1):
            if i not in vocab_en:
                vocab_en[i] = pad_token
        for i in range(1, max(minsize, max_de)+1):
            if i not in vocab_de:
                vocab_de[i] = pad_token
        for i in range(1, max(minsize, max_token_en)+1):
            if i not in tokenizedVocab_en:
                tokenizedVocab_en[i] = pad_token
        for i in range(1, max(minsize, max_token_de)+1):
            if i not in tokenizedVocab_de:
                tokenizedVocab_de[i] = pad_token
                
    print(f"length of vocab_en: {len(vocab_en)}, length of vocab_de: {len(vocab_de)}, length of tokenizedVocab_en: {len(tokenizedVocab_en)}, length of tokenizedVocab_de: {len(tokenizedVocab_de)}")
                
    return vocab_en, vocab_de, tokenizedVocab_en, tokenizedVocab_de
    
def DumpVocabToJSON(vocab, filename):
    with open(filename, 'w') as f:
        json.dump(vocab, f)
        

def AddInitEOSTokensToDataset(dataset, tokenizer, initToken=1, endToken=50256):
    initWord = tokenizer.decode([initToken])
    endWord = tokenizer.decode([endToken])
    for pair in dataset:
        pair[0] = pair[0] + initWord
        pair[1] = pair[1] + initWord
        pair[0] = pair[0] + endWord
        pair[1] = pair[1] + endWord
        
    return dataset


def DatasetToTokens(dataset, tokenizer, max_size=256, end_token="<|endoftext|>", batch_size=4):
    #convert a dataset to tokens
    batch_en = [tokenizer.encode(pair[0]) for pair in dataset]
    longest = max(batch_en, key=len)
    batch_en = [x + [0] * (len(longest) - len(x)) for x in batch_en]
    batch_en = torch.tensor(batch_en, dtype=torch.long)
    #tensor should be of shape (num batches, batch size, sequence length)
    #remove the last few to make it divisible by batch size
    if batch_en.shape[0] % batch_size != 0:
        batch_en = batch_en[:-(batch_en.shape[0] % batch_size)]
    batch_en = batch_en.reshape(-1, batch_size, batch_en.shape[1])
    
    batch_de = [tokenizer.encode(pair[1]) for pair in dataset]
    longest = max(batch_de, key=len)
    batch_de = [x + [0] * (len(longest) - len(x)) for x in batch_de]
    batch_de = torch.tensor(batch_de, dtype=torch.long)
    #tensor should be of shape (num batches, batch size, sequence length)
    if batch_de.shape[0] % batch_size != 0:
        batch_de = batch_de[:-(batch_de.shape[0] % batch_size)]
    batch_de = batch_de.reshape(-1, batch_size, batch_de.shape[1])
    
    
    assert len(batch_en) == len(batch_de)
    assert batch_en.shape[0] == batch_de.shape[0]
    #return the tokens
    return batch_en, batch_de

#tokenizer to convert txt to tokens
class Tokenizer():
    gpt = None
    def __init__(self):
        self.gpt = GPT2Tokenizer.from_pretrained('gpt2')
        
    def encode(self, text):
        return self.gpt.encode(text)
    
    def decode(self, tokens):
        return self.gpt.decode(tokens)

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, device=None):
        super().__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(hidden_dim, input_dim, device=self.device)
        self.linear2 = torch.nn.Linear(input_dim, hidden_dim, device=self.device)
        
        self.dropout = torch.nn.Dropout(dropout)
                
    def forward(self, x):
        x0 = self.dropout(torch.relu(self.linear1(x)))
        x1 = self.linear2(x0)
        return x1
    
class SimpleMultiHeadedAttention(torch.nn.Module):
    def __init__(self, hiddenDim, numHeads, dropout, device):
        super().__init__()
        self.device = device
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.head_dim = hiddenDim // numHeads
        
        self.dropout = torch.nn.Dropout(dropout)
        #self.fcQ = torch.nn.Linear(hiddenDim, hiddenDim, device=self.device)
        self.fcK = torch.nn.Linear(hiddenDim, hiddenDim, device=self.device)
        self.fcV = torch.nn.Linear(hiddenDim, hiddenDim, device=self.device)
        
        self.fc = torch.nn.Linear(hiddenDim, hiddenDim, device=self.device)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        #print(f"1query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        self.fcQ = torch.nn.Linear(self.hiddenDim, self.hiddenDim, device=self.device)
        Q = self.fcQ(query)
        #print(f"2aquery shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        K = self.fcK(key)
        #print(f"2bquery shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        V = self.fcV(value)
        #print(f"2cquery shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        
        Q = Q.view(batch_size, -1, self.numHeads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.numHeads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.numHeads, self.head_dim).permute(0, 2, 1, 3)
        #print(f"3query shape: {query.shape}, key shape: {key.shape}, value shape: {value.shape}")
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hiddenDim)
        x = self.fc(x)
        return x, attention


class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.dropout = torch.nn.Dropout(dropout)
        self.self_attentionLayerNorm = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.feedforwardLayerNorm = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.self_attention = SimpleMultiHeadedAttention(hidden_dim, num_heads, dropout, device)
        self.feedforward = FeedForward(hidden_dim, hidden_dim, dropout, device)
        
    def forward(self, input, inputMask=None):
        #input is a tensor of shape (batch_size, sequenceLength, hidden_dim)
        batch_size = input.shape[0]
        sequenceLength = input.shape[1]
        #print(f"input shape: {input.shape}, inputMask shape: {inputMask.shape}")
        
        #self attention with potential improved self attention
        if DO_IMPROVED_SELF_ATTENTION:
            _input = self.self_attentionLayerNorm(input)
            _input = self.self_attention(_input, _input, _input, inputMask)
            input = input + self.dropout(_input)
        else:
            _input, _ = self.self_attention(input, input, input, inputMask)
            input = self.self_attentionLayerNorm(input + self.dropout(_input))
        
        #feed forward
        _input = self.feedforward(input)
        x1 = self.feedforwardLayerNorm(input + self.dropout(_input))
        return x1
    
#all you need is attention encoder
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=6, num_heads=1, dropout=0.1, device=None, maxLen=256):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        #print(f"self.tokenEmbedding shape:, should be ({input_dim}, {hidden_dim})")
        self.positionalEncoding = torch.nn.Embedding(maxLen, hidden_dim, device=self.device)
        self.tokenEmbedding = torch.nn.Embedding(input_dim, hidden_dim, device=self.device)
        self.layers = torch.nn.ModuleList([EncoderLayer(hidden_dim, num_heads, dropout, device) for _ in range(num_layers)])
        self.scalingFactor = torch.sqrt(torch.FloatTensor([hidden_dim])).to(self.device)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, inputSequence, mask=None):
        #inputSequence is a tensor of shape (batch_size, sequenceLength)
        
        batch_size = inputSequence.shape[0]
        sequenceLength = inputSequence.shape[1]
        
        positons = torch.arange(0, sequenceLength)
        positons = positons.unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tokenEncoding = self.tokenEmbedding(inputSequence)
        tokenEncoding *= self.scalingFactor
        positionalEncoding = self.positionalEncoding(positons)
        x0 = self.dropout(tokenEncoding + positionalEncoding)
        #print(f"x0 shape: {x0.shape}, mask shape: {mask.shape}")
        for layer in self.layers:
            x0 = layer(x0, mask)
        return x0
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.dropout = torch.nn.Dropout(dropout)
        self.self_attentionLayerNorm = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.encoder_attentionLayerNorm = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.feedforwardLayerNorm = torch.nn.LayerNorm(hidden_dim, device=self.device)
        self.self_attention = SimpleMultiHeadedAttention(hidden_dim, num_heads, dropout, device)
        self.encoder_attention = SimpleMultiHeadedAttention(hidden_dim, num_heads, dropout, device)
        self.feedforward = FeedForward(hidden_dim, hidden_dim, dropout, device)
        
    def forward(self, target, encoded_input, input_mask=None, target_mask=None):
        #input is a tensor of shape (batch_size, sequenceLength, hidden_dim)
        batch_size = target.shape[0]
        sequenceLength = target.shape[1]
        #print(f"target shape: {target.shape}, encoded_input shape: {encoded_input.shape}, input_mask shape: {input_mask.shape}, target_mask shape: {target_mask.shape}")
        #self attention
        #_target, _ = self.self_attention(target, target, target, target_mask)
        #target = self.self_attentionLayerNorm(target + self.dropout(_target))
        #self attention with potential improved self attention
        if DO_IMPROVED_SELF_ATTENTION:
            _target = self.self_attentionLayerNorm(target)
            _target = self.self_attention(_target, _target, _target, target_mask)
            target = target + self.dropout(_target)
        else:
            _target, _ = self.self_attention(target, target, target, target_mask)
            target = self.self_attentionLayerNorm(target + self.dropout(_target))
        
        #encoder attention
        _target, attention = self.encoder_attention(target, encoded_input, encoded_input, input_mask)
        target = self.encoder_attentionLayerNorm(target + self.dropout(_target))
        
        #feed forward
        _target = self.feedforward(target)
        x2 = self.feedforwardLayerNorm(target + self.dropout(_target))
        return x2, attention
        
class Decoder(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=6, num_heads=1, dropout=0.1, device=None, maxLen=256):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.layers = torch.nn.ModuleList([DecoderLayer(hidden_dim, num_heads, dropout, device) for _ in range(num_layers)])
        
        self.linear = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.positionalEncoding = torch.nn.Embedding(maxLen, hidden_dim, device=self.device)
        self.tokenEmbedding = torch.nn.Embedding(output_dim, hidden_dim, device=self.device)
        self.scalingFactor = torch.sqrt(torch.FloatTensor([hidden_dim])).to(self.device)
        
    def forward(self, target, encoded_input, input_mask=None, target_mask=None):
        #target is a tensor of shape (batch_size, sequenceLength)
        #encoded_input is a tensor of shape (batch_size, sequenceLength, hidden_dim)
        batch_size = target.shape[0]
        sequenceLength = target.shape[1]
        #print(f"1x0 shape: {target.shape}, encoded_input shape: {encoded_input.shape}, input_mask shape: {input_mask.shape}, target_mask shape: {target_mask.shape}")
        positionEncoding = self.positionalEncoding(torch.arange(0, sequenceLength).unsqueeze(0).repeat(batch_size, 1).to(self.device))
        tokenEncoding = self.tokenEmbedding(target) * self.scalingFactor
        #print(f"2x0 shape: {target.shape}, encoded_input shape: {encoded_input.shape}, input_mask shape: {input_mask.shape}, target_mask shape: {target_mask.shape}")
        x0 = tokenEncoding + positionEncoding
        x0 = self.dropout(x0)
        for layer in self.layers:
            #print(f"3x0 shape: {x0.shape}, encoded_input shape: {encoded_input.shape}, input_mask shape: {input_mask.shape}, target_mask shape: {target_mask.shape}")
            x0, attention = layer(x0, encoded_input, input_mask=input_mask, target_mask=target_mask)
        output = self.linear(x0)
        #debugPrintOutputFromDecoder(output)
        return output, attention
    
def debugPrintOutputFromDecoder(output):
    print(f"output shape: {output.shape}")
    output = output[0, -1, 1:]
    output = torch.softmax(output, dim=0)
    print(f"new output shape: {output.shape}")
    tokenizer = Tokenizer()
    topk = torch.topk(output, 12)
    stringToPrint = ""
    for i in range(len(topk[0])):
        topk[0][i] = topk[0][i] / torch.sum(topk[0])
    
    #print(f"Top 10: {topk[0]} {topk[1]}")
    for i in range(len(topk[0])):
        index = topk[1][i].item()
        prob = topk[0][i].item()
        if index == None:
            continue
        try:
            word = tokenizer.decode(index)
        except:
            word = "err"
        stringToPrint += f"{index}-{word}: {prob:0.4f}, "    
    print(stringToPrint)
            
class AttentionTransformer(torch.nn.Module):
    def __init__(self, encoder, decoder, device=None):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.gptEOS = 50256
        self.gptPad = 50256
        
    def inputMask(self, input):
        #this mask only hides the padding
        mask = (input != self.gptEOS).unsqueeze(1).unsqueeze(2)
        return mask
        
    def targetMask(self, target):
        #this mask hides the padding and the future words
        targetLength = target.shape[1]
        batch_size = target.shape[0]
        padmask = (target != self.gptPad).unsqueeze(1).unsqueeze(2)
        attnmask = torch.tril(torch.ones((targetLength, targetLength), device=self.device)).bool()
        attnmask = attnmask.unsqueeze(0).repeat(batch_size, 1, 1)
        attnmask = attnmask & padmask
        return attnmask
        
    def forward(self, input, target):
        inputMask = self.inputMask(input)
       # outputMask = self.inputMask(target)
        targetMask = self.targetMask(target)
        #print(f"inputMask shape: {inputMask.shape}, targetMask shape: {targetMask.shape}, outputMask shape: {outputMask.shape}")
        
        encodedInput = self.encoder(input, mask=inputMask)
        output, attention = self.decoder(target, encodedInput, input_mask=inputMask, target_mask=targetMask)
        return output, attention
def params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initWeights(model):
    #use xavier uniform initialization
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(model.weight.data)
        
            
def trainModel(model, input, output, optimizer, criterion, device):
    model.train()
    total_loss = 0
    i = 0
    tstart = time.time()
    for data in zip(input, output):
        print(f"Batch {i+1}/{len(input)}", end="")
        tstartBatch = time.time()
        x = data[0].to(device)
        y = data[1].to(device)
        optimizer.zero_grad()
        #print(f"shape of x: {x.shape}, shape of y: {y.shape}")
        o, _ = model(x, y[:,:-1])
        o = o.contiguous().view(-1, o.shape[-1])
        
        #shift the target by one so that the decoder can predict the next token
        y = y[:,1:].contiguous().view(-1)
        loss = criterion(o, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        total_loss += loss.item()
        i += 1
        tendBatch = time.time()
        print(f" took {tendBatch-tstartBatch} seconds, loss: {loss.item()}")
    tend = time.time()
    print(f"Training Epoch took {tend-tstart} seconds")
    return total_loss / len(input)

def testModel(model, input, output, criterion, device):
    model.eval()
    total_loss = 0
    i = 0
    tstart = time.time()
    with torch.no_grad():
        for data in zip(input, output):
            x = data[0].to(device)
            y = data[1].to(device)
            o, _ = model(x, y[:,:-1])
            output_dim = o.shape[-1]
            o = o.contiguous().view(-1, output_dim)
            y = y[:,1:].contiguous().view(-1)
            loss = criterion(o, y)
            total_loss += loss.item()
    tend = time.time()
    print(f"Testing Epoch took {tend-tstart} seconds")
    return total_loss / len(input)


def inferenceSentance(model, tokenizer, sentance, device, maxLen=64, initToken=1, endToken=50256, batchSize=4, inferenceParams=None):
    if maxLen == 64:
        maxLen = inferenceParams["maxLen"]
    
    print(f"Running inference")
    model.eval()
    sentance = sentance.lower()
    sentance = sentance + "<|endoftext|>"
    tokens = tokenizer.encode(sentance)
    tokens.insert(0, initToken)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    inputMask = model.inputMask(tokens)
    inputTensor = torch.tensor(tokens).to(device)
    with torch.no_grad():
        sourceEncoded = model.encoder(inputTensor, mask=inputMask)
    #if batchSize > 1 and sourceEncoded.shape[0] == 1:
    #    sourceEncoded = sourceEncoded.repeat(batchSize, 1, 1)
    #    inputMask = inputMask.repeat(batchSize, 1, 1, 1)
    
    #print(type(maxLen))
    maxRetries = 5
    for retry in range(maxRetries):
        predictedTokens = [initToken]
        print(f"Inference retry: {retry+1}/{maxRetries}")
        for i in range(maxLen - 1):
            #print(f"Predicting token {i+1}/{maxLen}", end="")
            targetTensor = torch.tensor(predictedTokens).unsqueeze(0).to(device)
            #print(targetTensor)
            targetMask = model.targetMask(targetTensor)
            with torch.no_grad():
                output, attention = model.decoder(targetTensor, sourceEncoded, input_mask=inputMask, target_mask=targetMask)
            #predict the most likely token from the output
            #remove 0th possible tokens from the 2nd dimension
            output = output[0, -1, 1:]
            output = torch.softmax(output, dim=0)
            
            #print(output)
            #toptwenty = torch.topk(output, 20)
            #for i in range(len(toptwenty[0])):
            #    toptwenty[0][i] = toptwenty[0][i] / torch.sum(toptwenty[0])
            ##print(toptwenty)
            #formattedTwenty = ""
            #for i in range(len(toptwenty[0])):
            #    formattedTwenty += f"{[toptwenty[1][i].item()]}-{tokenizer.decode([toptwenty[1][i].item()])}: {100*toptwenty[0][i].item():0.4f}, "
            #print(f"Top 20: {formattedTwenty}")
            
            #topk = torch.topk(output, inferenceParams["topk"])
            #print(f"Top {inferenceParams['topk']}: {topk}")
            
            #sample from topk
            
            #(prob tensor, index tensor)
            #probList = topk[0].tolist()
            ##normalize probs
            #probList = [p / sum(probList) for p in probList]
            #indexList = topk[1].tolist()
            
            #print(f"Prob list: {probList}")
            #print(f"Index list: {indexList}")
            #wordlist = ""
            #for i in indexList:
            #    try:
            #        wordlist += f"{tokenizer.decode([i])}, "
            #    except:
            #        wordlist += f"{i}, "
            #print(f"Wordlist: {wordlist}")
            #
            ##pick a weighted random index
            #predicted = np.random.choice(indexList, 1, p=probList)[0]
            #print(f"Predicted: {predicted}")

            repPenalty = inferenceParams["repPenalty"]
            prevToken = predictedTokens[-1]
            #temp sampling
            predictions = np.asarray(output.cpu()).astype('float64')
            
            #apply rep penalty
            for i in range(len(predictions)):
                if predictions[i] == prevToken:
                    predictions[i] = predictions[i] * repPenalty
                    
            #apply logit biases list of dictionaries
            logitBias = inferenceParams["logitBias"]
            logitBiasObj = {}
            for i in range(len(logitBias)):
                tok = logitBias[i][0]
                probPenalty = logitBias[i][1]
                logitBiasObj[tok] = probPenalty
                
            

            predictions = np.exp(np.log(predictions) / inferenceParams["temp"])
            #print(predictions)

            for i in range(len(predictions)):
                if i in logitBiasObj:
                    predictions[i] = predictions[i] * logitBiasObj[i]

            #temp patch, replace nan with 0.01
            numThatIsntNan = 0
            averageProbs = 0

            for p in predictions:
                if not math.isnan(p):
                    numThatIsntNan += 1

            predictions[np.isnan(predictions)] = 0.00

            #averageProbs = np.sum(predictions) / len(predictions)

            ##normalize
            predictions = predictions / np.sum(predictions)
            probs = torch.from_numpy(predictions)
            #predicted = torch.multinomial(probs, num_samples=inferenceParams["topk"])
            #remove Nonetype and nan
            #for p in predicted:
            #    if p == None or math.isnan(p):
            #        predicted.remove(p)

            predicted = np.random.choice(len(probs), p=probs)
            ##predicted = np.random.choice(predicted, 1, p=predictions)
            #predicted = predicted[-1].item()
            ##print(predicted, end="")
            
            #print top 10 debug
            #topk = torch.topk(probs, 10)
            #stringToPrint = ""
            #for i in range(len(topk[0])):
            #    topk[0][i] = topk[0][i] / torch.sum(topk[0])
            #
            #print(f"Top 10: {topk[0]} {topk[1]}")
            #for i in range(len(topk[0])):
            #    index = topk[1][i].item()
            #    prob = topk[0][i].item()
            #    if index == None:
            #        continue
            #    try:
            #        word = tokenizer.decode(index)
            #    except:
            #        word = "err"
            #    stringToPrint += f"{index}-{word}: {prob:0.4f}, "
            #print(f"Top 10: {stringToPrint}")
            


            predictedDecodedToken = None
            try:
                predictedDecodedToken = tokenizer.decode([predicted])
                print(predictedDecodedToken, end="")
            except:
                print(f"error decoding token")
                pass

            if predictedDecodedToken != None:
                predictedTokens.append(predicted)
            else:
                predictedTokens.append(0)


            if predicted == endToken or predictedDecodedToken == inferenceParams["eos"]:
                break 
            
            
            #if i % 32 == 0:
#
            #    predictedSentance = tokenizer.decode(predictedTokens)
            #    predictedSentance = cleanNonReadableText(predictedSentance)
            #    print(f"Output: {predictedSentance}")
        if len(predictedTokens) >= inferenceParams["minLen"]:
            print(f"\nLength of predicted tokens: {len(predictedTokens)}, greater than minLen: {inferenceParams['minLen']}")
            break
    predictedSentance = tokenizer.decode(predictedTokens)
    predictedSentance = cleanNonReadableText(predictedSentance)
    return predictedSentance, attention

#borrowed
def display_attention(sentence, translation, attention, n_heads = 4, n_rows = 1, n_cols = 4):
    
    assert n_rows * n_cols == n_heads
    print(f"a shape: {attention.shape}")
    #get first batch
    attention = attention[0]
    
    fig = plt.figure(figsize=(25,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        print(f"a2 shape: {_attention.shape}")
        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], rotation=45)
        ax.set_yticklabels(translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
    
def cleanNonReadableText(text):
    cleanText = text.strip()
    cleanText = unidecode(cleanText)
    for c in cleanText:
        if c not in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!'\"<>|[]{}()":
            cleanText = cleanText.replace(c, "")
    cleanText = cleanText.replace("<|endoftext|>", "")
    cleanText = cleanText.replace("<|startoftext|>", "")
    cleanText = cleanText.replace("�", "")
    #capitalize the first letter
    cleanText = cleanText.strip()
    cleanText = cleanText[0].upper() + cleanText[1:]
    return cleanText

def calcBLEU(inferenceresults):
    #inferenceresults is a list of lists of [predicted, truth]
    #calculate the BLEU score
    #https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    references = []
    candidates = []
    for result in inferenceresults:
        references.append(result[1])
        candidates.append(result[0])
    score = corpus_bleu(references, candidates)
    print(f"BLEU score: {score}")
    return score



def calcBLEUWrapper(model, dataset_en, dataset_de, inferenceParams, device):
    tokenizer = Tokenizer()
    infResults = []
    for i in range(len(dataset_en)):
        print(f"Calculating BLEU for {i+1}/{len(dataset_en)}")
        osentence = tokenizer.decode(dataset_en[i][0])
        desentance = tokenizer.decode(dataset_de[i][0])
        translation, attention = inferenceSentance(
        model=model, tokenizer=tokenizer, sentance=osentence, device=device, maxLen=inferenceParams["maxLen"], inferenceParams=inferenceParams)
        translation = cleanNonReadableText(translation)
        
        #split into list of words
        translation = translation.split()
        desentance = desentance.split()
        print(f"Length of translation: {len(translation)}, length of desentance: {len(desentance)}")
        infResults.append([translation, desentance])
        
    return calcBLEU(infResults)

def writePickle(obj, filename):
    root = 'picklejar'
    if not os.path.exists(root):
        os.makedirs(root)
        
    filename = os.path.join(root, filename)
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
    

def main():
    print(f"Running Preinit")
    device = preinit()
    PAD_TOKEN = 50256
    EOS_TOKEN = 50256
    INIT_TOKEN = 1
    model = None
    InferenceOnly = False
    SaveModel = True
    SaveModelPath = "model.pt"
    LoadModelPath = "Goodmodel.pt"
    
    trainPairs= loadDatasets(True)
    #only use 50 pairs for now
    trainPairs = trainPairs[:24*8]
    print(f"Loaded {len(trainPairs)} pairs")
    
    global DO_IMPROVED_SELF_ATTENTION
    DO_IMPROVED_SELF_ATTENTION = False
    hidden_dim = 512
    num_layers = 6
    dropoutEncoder = 0.1
    dropoutDecoder = 0.1
    num_heads = 8
    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    TrainTestFraction = 0.4
    tokenizer = Tokenizer()
    trainPairs = AddInitEOSTokensToDataset(trainPairs, tokenizer, initToken=INIT_TOKEN, endToken=EOS_TOKEN)
    
    batch_en, batch_de = DatasetToTokens(trainPairs, tokenizer, batch_size=BATCH_SIZE)
    
    batch_en_test, batch_en_train = batch_en[:math.floor(len(batch_en) * TrainTestFraction)], batch_en[math.floor(len(batch_en) * TrainTestFraction):]
    batch_de_test, batch_de_train = batch_de[:math.floor(len(batch_de) * TrainTestFraction)], batch_de[math.floor(len(batch_de) * TrainTestFraction):]
    print(f"Train has a length of {len(batch_en_train)} batches, test has a length of {len(batch_en_test)} batches")
    vocab_en, vocab_de, tokenizedvocab_en, tokenizedvocab_de = countVocab(trainPairs, tokenizer)
    DumpVocabToJSON(vocab_en, "vocab_en.json")
    DumpVocabToJSON(vocab_de, "vocab_de.json")
    DumpVocabToJSON(tokenizedvocab_en, "tokenizedvocab_en.json")
    DumpVocabToJSON(tokenizedvocab_de, "tokenizedvocab_de.json")
    print(f"There are {len(batch_en)} batches in the training set")
    #print(batch_en.shape)
    input_dim = len(vocab_en)
    output_dim = len(vocab_de)
    
    
    
    #test
    LR = 0.0005
    testPhrase = "Hello, my name is John. My favorite place in the world is the beach."
    truthPhrase = translated = GoogleTranslator(source='auto', target='de').translate(testPhrase)
    tokensTestPhrase = tokenizer.encode(testPhrase)
    #print(tokensTestPhrase)
    
    #inference params
    maxLen = 64
    temp = 0.7
    topk = 12
    logitBiases = list()
    logitBiases.append((9971, 0.2))
    logitBiases.append((4582, 0.2))
    logitBiases.append((4861, 0.2))
    logitBiases.append((5745, 0.2))
    logitBiases.append((4655, 0.2))
    logitBiases.append((2149, 0.2))
    InferenceParams =  {"maxLen": maxLen, "temp": temp, "topk": topk, "eos": tokenizer.decode([EOS_TOKEN]),
                        "init": tokenizer.decode([INIT_TOKEN]), "pad": tokenizer.decode([PAD_TOKEN]),"minLen": 10,
                        "repPenalty": 0.2, "logitBias": logitBiases}
    
    #tracking
    lossByEpoch = []
    timeTrainingByEpoch = []
    inferenceResultsByEpoch = []
    totalTimeStart = time.time()
    if InferenceOnly:
        #load model.pt instead of training
        model = torch.load(LoadModelPath)
        SaveModel = False
        print("Model loaded from file")        
    else:
        encoder = Encoder(input_dim, hidden_dim, num_layers, num_heads, dropout=dropoutEncoder, device=device)
        decoder = Decoder(output_dim, hidden_dim, num_layers, num_heads, dropout=dropoutDecoder, device=device)
        model = AttentionTransformer(encoder, decoder, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        
        print(f"Model has {params(model)} parameters")
        model.apply(initWeights)
        #inference once at beginning
        translation, attention = inferenceSentance(
        model=model, tokenizer=tokenizer, sentance=testPhrase, device=device, maxLen=maxLen, inferenceParams=InferenceParams)
        print(f"--Starting INFERENCE--")
        print(f"Input: {testPhrase}")
        print(f"Output: {translation}")
        translation = cleanNonReadableText(translation)
        print(f"Cleaned Output: {translation}")
        print(f"Truth: {truthPhrase}")
        inferenceResultsByEpoch.append([translation, truthPhrase])
        print(f"----------------")
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            start = time.time()
            train_loss = trainModel(model, batch_en_train, batch_de_train, optimizer, criterion, device)
            test_loss = testModel(model, batch_en_test, batch_de_test, criterion, device)
            print(f"Train Loss: {train_loss}, Test Loss: {test_loss}")
            end = time.time()
            print(f"Epoch took {end-start} seconds")
            translation, attention = inferenceSentance(
        model=model, tokenizer=tokenizer, sentance=testPhrase, device=device, maxLen=maxLen, inferenceParams=InferenceParams)
            print(f"--INFERENCE--")
            print(f"Input: {testPhrase}")
            print(f"Output: {translation}")
            translation = cleanNonReadableText(translation)
            print(f"Cleaned Output: {translation}")
            print(f"Truth: {truthPhrase}")
            inferenceResultsByEpoch.append([translation, truthPhrase])
            print(f"----------------")
            
            #save model every 5 epochs
            if epoch % 5 == 0:
                torch.save(model.state_dict(), SaveModelPath + f"_{epoch}.pt")
                print("Model saved")
        
            #save tracking
            lossByEpoch.append([train_loss, test_loss])
            timeTrainingByEpoch.append(end-start)
        
            
        #save the model
        if SaveModel:
            torch.save(model.state_dict(), SaveModelPath)
            print("Model saved")
        print(f"Train Loss: {train_loss}, Test Loss: {test_loss}")
        print(f"Took {end-start} seconds")
        
        #save results pickle
        writePickle(lossByEpoch, "lossByEpoch.pickle")
        writePickle(timeTrainingByEpoch, "timeTrainingByEpoch.pickle")
        writePickle(inferenceResultsByEpoch, "inferenceResultsByEpoch.pickle")
    totalTimeEnd = time.time()
    #run final inference
    translation, attention = inferenceSentance(
        model=model, tokenizer=tokenizer, sentance=testPhrase, device=device, maxLen=maxLen, inferenceParams=InferenceParams)
    
    print(f"--FINAL INFERENCE--")
    print(f"Input: {testPhrase}")
    print(f"Output: {translation}")
    translation = cleanNonReadableText(translation)
    print(f"Cleaned Output: {translation}")
    print(f"Truth: {truthPhrase}")
    print(f"----------------")
    print(f"Params of model: {params(model)}")
    print(f"Time Taken: {totalTimeEnd - totalTimeStart}")
    
    bleu = calcBLEUWrapper(model, batch_en_test, batch_de_test, InferenceParams, device)
    print(f"BLEU score: {bleu}")
    
    #pickle time taken
    writePickle(totalTimeEnd - totalTimeStart, "totalTime.pickle")
    writePickle(bleu, "bleu.pickle")


if __name__ == '__main__':
    main()