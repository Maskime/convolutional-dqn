import os
from collections import namedtuple
from typing import Dict, Any, Callable

from alien_gym import Config

CommandLineConfig = namedtuple('CommandLineConfig', ['config', 'checkpoint', 'nb_config'])


def extract_commandlineconfig(default_config: Config, command_line: Dict[str, Any], video_callable: Callable) -> Config:
    command_line = dict((k, v) for k, v in command_line.items() if v is not None)

    default_config_dict = default_config._asdict()
    default_config_dict.update(command_line)
    if command_line['with_record']:
        default_config_dict['record'] = video_callable
    del default_config_dict['with_record']
    checkpoint_file = None
    if 'checkpoint' in command_line and command_line['checkpoint']:
        checkpoint_file = command_line['checkpoint']
        if not os.path.isfile(checkpoint_file):
            raise Exception('Provided checkpoint file [{}] can not be read'.format(checkpoint_file))
        del default_config_dict['checkpoint']
    nb_configurations = default_config_dict['nb_config']
    del default_config_dict['nb_config']
    return CommandLineConfig(config=Config(**default_config_dict), checkpoint=checkpoint_file,
                             nb_config=nb_configurations)
