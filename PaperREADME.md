# How to run

Main file (main.py) - Just run with any python interpreter, make sure all imports are installed, no params needed

Cuda check file (checkCUDA.py) - Just run with any python interpreter, make sure all imports are installed, no params needed

Model moving helper file (replaceGoodModel.py) - Just run with any python interpreter, make sure all imports are installed, no params needed

Test Tokenizer file (tokenizertest.py) - Just run with any python interpreter, make sure all imports are installed, no params needed

Graph Generation file (unpickleData.py) - Just run with any python interpreter, make sure all imports are installed, no params needed This one must be run after the main file.

# Code Attribution

## Copied files

No code files have been directly copied

## Modified files

main.py heavily reflects the pytorch reimplementation found here: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb

Except for a few helpers, all code was written from scratch while following the above link as a tutorial. 

## New files

Except for main.py, all other helper files are 100% written by Rex Wang

# Datasets

The Flores 200 dataset was used for this project. It is a translation dataset that includes 2000 sentances of the same content in a variety of languages. It was obtained on their public site found here:

https://github.com/facebookresearch/flores/blob/main/flores200/README.md

Downloaded with:

wget --trust-server-names https://tinyurl.com/flores200dataset