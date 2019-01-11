import argparse
import csv
import datetime
from random import randint, uniform

from alien_gym import Config, AlienGym, AlienGymResult
from cdqn_logging import cdqn_logger
from commandline_parameters import extract_commandlineconfig, CommandLineConfig

nb_runs = 1  # number of runs for each configuration
configs = []
# Need some consistency between test
default_nb_epoch = 50
default_imagesize = '80x80'


def video_callable(episode_id):
    return True


default_config = Config(
    nb_epoch=default_nb_epoch, with_crop=True, with_color_pre=True, image_size=default_imagesize, nb_games=3,
    softmax_temp=1.0, n_step=10, memory_capacity=8000, optimizer_lr=0.001, gamma=0.99, record=None, is_train=True)

parser = argparse.ArgumentParser(description='Run CDQN on AlienV0 OpenAI Gym')
parser.add_argument('--image-size', dest='image_size',
                    help='Image size the CNN will be working with (format : WxH)')
parser.add_argument('--nb-epoch', dest='nb_epoch', type=int,
                    help='''Number of epoch that will be run. If the --checkpoint option is set, 
                    will resume training for this amount of epoch''')
parser.add_argument('--without-crop', dest='with_crop', action='store_false',
                    help='Disable the image cropping when pre-processing the image')
parser.add_argument('--without-color_pre', dest='with_color_pre', action='store_false',
                    help='Disable the image color alteration when pre-processing the image')
parser.add_argument('--nb-games', dest='nb_games', type=int,
                    help='''Number of games that will be play for 1 epoch. If the --play flag is set, this value will 
                    be used for the number of games played with the set model.''')
parser.add_argument('--softmax-temp', dest='softmax_temp', type=float,
                    help='Constant value that will be applied to the softmax function')
parser.add_argument('--memory-capacity', dest='memory_capacity', type=int,
                    help='Number of n-steps that will be saved for the n-step Q-Learning')
parser.add_argument('--optimizer-lr', dest='optimizer_lr', type=float, help='Learning rate for the optimizer')
parser.add_argument('--n-step', dest='n_step', type=int, help='Number of steps before backprop is done on the CNN')
parser.add_argument('--gamma', dest='gamma', type=float, help='Living penalty')
parser.add_argument('--record', dest='with_record', action='store_true', help='Should we record the games ?')
parser.add_argument('--nb-config', dest='nb_config', type=int,
                    help='If specify will generate nb_config random configurations', default=0)
parser.add_argument('--checkpoint', dest='checkpoint', type=str, help='Will resume training from a checkpoint file')
parser.add_argument('--play', dest='is_train', action='store_false',
                    help='''No training happening, you MUST provide a checkpoint file with this option. 
                    If you set the --nb-games option this value will be use for the number of games played with the
                    set model''')

commandline_config: CommandLineConfig = extract_commandlineconfig(default_config=default_config,
                                                                  command_line=vars(parser.parse_args()),
                                                                  video_callable=video_callable)
nb_configurations = commandline_config.nb_config
configs.append(commandline_config.config)

for i in range(0, nb_configurations):
    configs.append(
        Config(
            image_size=default_imagesize,
            nb_epoch=default_nb_epoch,
            with_crop=bool(randint(0, 1)),
            with_color_pre=bool(randint(0, 1)),
            nb_games=randint(2, 10),
            softmax_temp=uniform(0.1, 1.5),
            memory_capacity=randint(50, 10000),
            optimizer_lr=uniform(0.01, 0.001),
            n_step=randint(5, 200),
            gamma=uniform(1., 0.01),
            record=video_callable,
            is_train=True
        ))

alien_gym = AlienGym()
run_number = 0
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_dict(result: AlienGymResult):
    config_map = result.config._asdict()
    general_map = result._asdict()
    del general_map['config']
    return {**config_map, **general_map}


stat_filename = 'run_stats_{}.csv'.format(now)
dummy = AlienGymResult(config=configs[0], final_mean=0, min=0, max=0, total_time=0, run_name='dir')
fieldsname = get_dict(dummy).keys()
with open(stat_filename, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldsname)
    writer.writeheader()

cdqn_logger.info('[{}] config to run'.format(len(configs)))
for config in configs:
    for i in range(0, nb_runs):
        cdqn_logger.info('---------Starting run {}---------'.format(run_number))
        result: AlienGymResult = alien_gym.run(config=config, run_number=run_number)
        with open(stat_filename, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=get_dict(dummy).keys())
            writer.writerow(get_dict(result=result))
        run_number += 1
