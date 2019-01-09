from alien_gym import Config, AlienGym
from cdqn_logging import cdqn_logger

nb_runs = 3  # number of runs for each configuration

configs = [Config(
    nb_epoch=1,
    with_crop=True,
    with_color_pre=True,
    image_size='80x80',
    nb_games=3,
    softmax_temp=0.5,
    n_step=10,
    memory_capacity=1000,
    optimizer_lr=0.001
),
    Config(
        nb_epoch=1,
        with_crop=True,
        with_color_pre=True,
        image_size='80x80',
        nb_games=3,
        softmax_temp=0.5,
        n_step=15,
        memory_capacity=1000,
        optimizer_lr=0.001
    )
]

alien_gym = AlienGym()
run_number = 0
for config in configs:
    for i in range(0, nb_runs):
        cdqn_logger.info('---------Starting run {}---------'.format(run_number))
        result = alien_gym.run(config=config, run_number=run_number)
        print(result)
        run_number += 1
