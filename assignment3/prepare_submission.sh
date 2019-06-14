#!/bin/bash

# Script for preparing a submission for the homework

echo "Preparing Submission. Please hold on."

# Check to make sure all of the files exist
if [ ! -d "models/" ]; then
   echo -e "\e[31mWARNING!!! models/ folder does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "1 Language Modeling.ipynb" ]; then
   echo -e "\e[31mWARNING!!! '1 Language Modeling.ipynb' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "2 Summarization.ipynb" ]; then
   echo -e "\e[31mWARNING!!! '2 Summarization.ipynb' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "transformer_layers.py" ]; then
   echo -e "\e[31mWARNING!!! 'transformer_layers.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "transformer_attention.py" ]; then
   echo -e "\e[31mWARNING!!! 'transformer_attention.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "transformer.py" ]; then
   echo -e "\e[31mWARNING!!! 'transformer.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi

zip -r submission.zip "models/" "1 Language Modeling.ipynb" "2 Summarization.ipynb" "transformer_layers.py" "transformer_attention.py" "transformer.py"

echo "Submission complete. Please submit the \"submission.zip\" file which has been created."