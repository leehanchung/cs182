# Berkeley CS182/282 Designing, Visualizing and Understanding Deep Neural Networks
Self study on CS182/CS282 - Designing, Visualizing and Understanding Deep Neural Networks (Spring 2019) @ UC Berkeley. Includes assignments, lecture slides, and lecture notes. Solutions passed all the self-contained unit tests but were not submitted using student-only submission system.

## Assignment1: :heavy_check_mark:
Based on Stanford CS 231n assignment 2. Only Python 2.7 supported. Migrated and completed using Python 3.7.

- Implenentation of fully connected deep neural network for classification using numpy only.

## Assignment2: :heavy_check_mark:
Based on Stanford CS 231n assignment 3. Python 3.5 and 3.6 supported. Done using Python 3.7. Network visualization and style transfer done using `pytorch 1.2`. Also implemented GAN notebook.

- Implementation of image captioning neural network using numpy only.
- Implementation of saliency map using Pytorch.
- Implementation of image style transfer using Pytorch.
- Implemnentation of GAN and DCGAN using using Pytorch.

## Assignment3: :heavy_check_mark:
Done using Python 3.7 and tensorflow 2.0 in `tensorflow.compat.v1` mode. Got 4.54 validation loss with default transformer hyper params vs. possible <= 4.5 indicated in the assignment notebook. Batch size limited by GPU memory.

- Implementation of Transformer from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), using Tensorflow 1.

## Assignment4: :heavy_check_mark:
Based on assignment 2 and 3  of CS294-112 Deep Reinforcement Learning at UC Berkeley. Python 3.5 and 3.6 supported,`tensorflow 1.10` code base. 

Done using Python 3.7 and Tensorflow 2.0 in `tensorflow.compat.v1` mode. Migraded codes in `train_dqn.py` from `Tensorflow 1.10` to `Tensorflow 1.15`. OpenAI Gym FFMPEG [issue](https://github.com/openai/gym/issues/35) prevented pong from training, causing ```ERROR: VideoRecorder encoder exited with status 1```. 

```
dd if=/dev/zero bs=750000 count=50 | ffmpeg -nostats -loglevel error -y -r 60 -f rawvideo -s:│Requirement already satisfied, skipping upgrade: six in /home/han/.virtualenvs/assignment4-RMXgM69L/lib/python3.7/site-packages (from a
v 500x500 -pix_fmt 'rgb24' -i /dev/stdin -vcodec libx264 -pix_fmt yuv420p /tmp/foo.mp4
Unknown encoder 'libx264'
dd: error writing 'standard output': Broken pipe
2+0 records in
1+0 records out
815536 bytes (816 kB, 796 KiB) copied, 0.00551571 s, 148 MB/s 
```

- Implementation of Vanilla Policy Gradient, DQN, DDQN.
