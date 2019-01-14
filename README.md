# Convolutional-dqn
Convolutional DQN that plays the Alien-V0 OpenAI Gym environment.

## Installation

Everything can be installed from pip.
For any missing package on your system (not the ones from python because those ones are handled with
`pip`) please refer to the openAI Gym documentation [Installing Everything](https://github.com/openai/gym#installing-everything)

If you have a system that has CUDA support, you need to install the appropriate PyTorch package, after that
the script will pick it up if it is available (`torch.cuda.is_available()`) otherwise it will use the
CPU (which is considerably slower.)

## What does it do ?

This is an implementation of que Deep Q-Learning with Experience Replay algorithm with a very basic Convolutional Neural Network.
It also has an eligibility trace implementation.

The OpenAI Gym env provide the current image which is sent to the CNN.

An epoch is defined by the number of games that will be played before the CNN weights are adjusted in
accordance to MSE Loss function.

The `data` dir will contain the file related to the current run:
  + Videos if asked to record them
  + `.log` files that contains information about the run.

## Available option at launch
```
$ python src/main.py -h
usage: main.py [-h] [--image-size IMAGE_SIZE] [--nb-epoch NB_EPOCH]
               [--without-crop] [--without-color_pre] [--nb-games NB_GAMES]
               [--softmax-temp SOFTMAX_TEMP]
               [--memory-capacity MEMORY_CAPACITY]
               [--optimizer-lr OPTIMIZER_LR] [--n-step N_STEP] [--gamma GAMMA]
               [--record] [--nb-config NB_CONFIG] [--checkpoint CHECKPOINT]
               [--play]

Run CDQN on AlienV0 OpenAI Gym

optional arguments:
  -h, --help            show this help message and exit
  --image-size IMAGE_SIZE
                        Image size the CNN will be working with (format : WxH)
  --nb-epoch NB_EPOCH   Number of epoch that will be run. If the --checkpoint
                        option is set, will continue training for this amount
                        of epoch
  --without-crop        Disable the image cropping when pre-processing the
                        image
  --without-color_pre   Disable the image color alteration when pre-processing
                        the image
  --nb-games NB_GAMES   Number of games that will be play for 1 epoch. If the
                        --play flag is set, this value will be used for the
                        number of games played with the set model.
  --softmax-temp SOFTMAX_TEMP
                        Constant value that will be applied to the softmax
                        function
  --memory-capacity MEMORY_CAPACITY
                        Number of n-steps that will be saved for the n-step
                        Q-Learning
  --optimizer-lr OPTIMIZER_LR
                        Learning rate for the optimizer
  --n-step N_STEP       Number of steps before backprop is done on the CNN
  --gamma GAMMA         Living penalty
  --record              Should we record the games ?
  --nb-config NB_CONFIG
                        If specify will generate nb_config random
                        configurations
  --checkpoint CHECKPOINT
                        Will resume training from a checkpoint file
  --play                No training happening, you MUST provide a checkpoint
                        file with this option. If you set the --nb-games
                        option this value will be use for the number of games
                        played with the set model
```

## Generated files

### `data/[run timestamp]/[run timestamp].log`

This files contains information about the run, namely :
  + min, max, average of played games during the epoch
  + Overall average
  
This file can be parsed using the `src/activity_log_parser.py` script which will generate a CSV file
with the following structure :

The first line : The configuration that was used for the run
Then: 1 line by epoch with min, max and average for the played games and the overall score average

### `./run_stats_[date_time].csv`

With the `--nb-config` option you can generate random configurations, the result of the runs for each
configuration will be stored in this file with the following information :
`nb_epoch,with_crop,with_color_pre,image_size,nb_games,softmax_temp,n_step,memory_capacity,optimizer_lr,gamma,record,is_train,final_mean,min,max,total_time,run_name`


## General thoughts
The average score I was able to obtain was between 200 and 300 with some max going to 1700 but they don't happen
that often, so I have to treat them more as random happening than actual learning.

What I think is happening : The pills that the player can pick-up are blinking on the screen which
makes harder for the current CNN to find the correlation between the pills and the reward.
What I'd like to try to best those score :
 + Better image pre-processing, I was thinking about summing 3 states so that at least during 1 frame
 all the game elements would be there, but that would create some weird pixel considering that the aliens 
 are not idle while I'd summed up the frames
 + Better CNN : I think the current CNN is too general to handle this problem. I would add the following convolution:
   + One for the pills
   + One for the player
   + One for the aliens that are moving around
 + Linear regression to max the configuration : There are few parameters that can be tuned to make
 the learning process better. My first intent with the `run_stats*` files was to use this as a dataset
 with a LR to obtain the best configuration.

### Data exploration
During the development of this I did do some few runs.

You can find the files that were generated and exploited in the following [Google Sheet](https://docs.google.com/spreadsheets/d/13xyjjkaEge6XJu5EaUd0tXLBM6KL7g0wmfmXvZVCzyc/edit?usp=sharing)