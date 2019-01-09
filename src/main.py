from alien_gym import Config, AlienGym

nb_runs = 3  # number of runs for each configuration

config = Config(
    nb_epoch=20,
    with_crop=True,
    with_color_pre=True,
    image_size='80x80',
    nb_games=2,
    softmax_temp=0.5,
    n_step=10,
    memory_capacity=1000,
    optimizer_lr=0.001
)

alien_gym = AlienGym()
alien_gym.run(config=config)