files="LSTM_Captioning.ipynb
NetworkVisualization-PyTorch.ipynb
NetworkVisualization-TensorFlow.ipynb
RNN_Captioning.ipynb
StyleTransfer-PyTorch.ipynb
StyleTransfer-TensorFlow.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required notebook $file not found."
        exit 0
    fi
done


rm -f assignment2.zip
zip -r assignment2.zip . -x "*.git" "*deeplearning/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements*.txt" ".env/*" "*.pyc" "*deeplearning/build/*"
