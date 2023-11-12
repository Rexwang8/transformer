#test tokenize a string with gpt2
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import random
import time
import os
import sys

#load model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# string
string = "<|startoftext|>"
# tokenize
tokenized_string = tokenizer.encode(string)
print(tokenized_string)