import csv
import datetime
from random import randint, uniform

from alien_gym import Config, AlienGym, AlienGymResult
from cdqn_logging import cdqn_logger

nb_runs = 1  # number of runs for each configuration
nb_configurations = 1000
configs = []
# Need some consistency between test
default_nb_epoch = 50
default_imagesize = '80x80'
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
