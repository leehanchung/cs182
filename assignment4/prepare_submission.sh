#!/bin/bash

# Script for preparing a submission for the homework

echo "Preparing Submission. Please hold on."

# Check to make sure all of the files exist
if [ ! -f "Part01_Policy-Gradients.ipynb" ]; then
   echo -e "\e[31mWARNING!!! 'Part01_Policy-Gradients.ipynb' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "Part02_Deep-Q-Networks.ipynb" ]; then
   echo -e "\e[31mWARNING!!! 'Part02_Deep-Q-Networks.ipynb' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -d "data_pg" ]; then
   echo -e "\e[31mWARNING!!! 'data_pg/' directory does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -d "data_dqn" ]; then
   echo -e "\e[31mWARNING!!! 'data_dqn/' directory does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "train_dqn.py" ]; then
   echo -e "\e[31mWARNING!!! 'train_dqn.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "train_pg.py" ]; then
   echo -e "\e[31mWARNING!!! 'train_pg.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "dqn.py" ]; then
   echo -e "\e[31mWARNING!!! 'dqn.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "logz.py" ]; then
   echo -e "\e[31mWARNING!!! 'logz.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "dqn_utils.py" ]; then
   echo -e "\e[31mWARNING!!! 'dqn_utils.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -f "atari_wrappers.py" ]; then
   echo -e "\e[31mWARNING!!! 'atari_wrappers.py' does not exist. It doesn't look like this submission is complete\e[39m."
fi
if [ ! -d "files" ]; then
   echo -e "\e[31mWARNING!!! 'files/' does not exist. It doesn't look like this submission is complete\e[39m."
fi

zip -r submission.zip "Part01_Policy-Gradients.ipynb" "Part02_Deep-Q-Networks.ipynb" "data_pg/" "data_dqn/" "train_dqn.py" "train_pg.py" "dqn.py" "logz.py" "dqn_utils.py" "atari_wrappers.py" "files/"

echo "Submission complete. Please submit the \"submission.zip\" file which has been created."
