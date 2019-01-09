import csv
import datetime

from alien_gym import Config, AlienGym, AlienGymResult
from cdqn_logging import cdqn_logger

nb_runs = 3  # number of runs for each configuration

configs = [
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        memory_capacity=1000, optimizer_lr=0.001,
        n_step=10
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        memory_capacity=1000, optimizer_lr=0.001,
        n_step=20
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        memory_capacity=1000, optimizer_lr=0.001,
        n_step=40
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        memory_capacity=1000, optimizer_lr=0.001,
        n_step=80
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        n_step=10, optimizer_lr=0.001,
        memory_capacity=2000
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        n_step=10, optimizer_lr=0.001,
        memory_capacity=4000
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, softmax_temp=0.5,
        n_step=10, optimizer_lr=0.001,
        memory_capacity=8000
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, memory_capacity=1000,
        n_step=10, optimizer_lr=0.001,
        softmax_temp=0.1
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, memory_capacity=1000,
        n_step=10, optimizer_lr=0.001,
        softmax_temp=0.2
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, memory_capacity=1000,
        n_step=10, optimizer_lr=0.001,
        softmax_temp=0.4
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, memory_capacity=1000,
        n_step=10, optimizer_lr=0.001,
        softmax_temp=0.8
    ),
    Config(
        nb_epoch=20, with_crop=True, with_color_pre=True, image_size='80x80', nb_games=3, memory_capacity=1000,
        n_step=10, optimizer_lr=0.001,
        softmax_temp=1.6
    )

]

alien_gym = AlienGym()
run_number = 0
now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def get_dict(result: AlienGymResult):
    config_map = result.config._asdict()
    general_map = result._asdict()
    del general_map['config']
    return {**config_map, **general_map}


stat_filename = 'run_stats_{}.csv'.format(now)
dummy = AlienGymResult(config=configs[0], final_mean=0, min=0, max=0, total_time=0, videos_dir='dir')
fieldsname = get_dict(dummy).keys()
with open(stat_filename, 'w') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldsname)
    writer.writeheader()

for config in configs:
    for i in range(0, nb_runs):
        cdqn_logger.info('---------Starting run {}---------'.format(run_number))
        result: AlienGymResult = alien_gym.run(config=config, run_number=run_number)
        with open(stat_filename, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=get_dict(dummy).keys())
            writer.writerow(get_dict(result=result))
        run_number += 1
