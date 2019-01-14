import argparse
import csv
import glob
import os
from collections import namedtuple

csvs_dir = os.path.join(os.path.dirname(__file__), '..', 'csvs')
glob_pattern = '{}/**/*.log'.format(csvs_dir)

log_files = [os.path.realpath(path) for path in glob.glob(glob_pattern)]

stats_fields = ['epoch', 'avg_score', 'min_score', 'max_score', 'global_avg']
AlienGymRunStats = namedtuple('AlienGymRunStats', stats_fields)

parser = argparse.ArgumentParser('Parse run activity logs')
parser.add_argument('--file', dest='file', type=str)
args = parser.parse_args()
if args.file is not None:
    log_files = [args.file]

for path_activitylog in log_files:
    print('reading ', path_activitylog)
    stats_filecontent = []
    with open(path_activitylog, 'r') as activity_log:
        current_epoch = 0
        current_gymstat = None
        for line in activity_log:
            time, level, run_name, message = line.split('::')
            if 'Using config : ' in message:
                stats_filecontent.append(message[16:].replace(',', ' '))
            elif 'Starting epoch ' in message:
                current_epoch = int(message[16:])
            elif 'Games done in : ' in message:
                game_time, avg_string, min_string, max_string = message[17:].split(',')
                # avg score
                game_avg = float(avg_string.strip()[9:])
                # min 90.0
                game_min = float(min_string.strip()[3:])
                # max 280.0
                game_max = float(max_string.strip()[3:])
            elif 'Epoch: {}'.format(current_epoch) in message:
                global_avg_string = message.split(',')[1].strip()
                # Average reward: 177.77777777777777
                global_avg = float(global_avg_string[15:])
                stats_filecontent.append(
                    AlienGymRunStats(epoch=current_epoch, avg_score=game_avg, min_score=game_min, max_score=game_max,
                                     global_avg=global_avg)._asdict())

    if len(stats_filecontent) < 20:
        print('File {} not intresting'.format(path_activitylog))
        continue

    dirname = os.path.dirname(path_activitylog)
    stat_csv = os.path.join(dirname, '..', '{}.csv'.format(os.path.basename(dirname)))
    with open(stat_csv, 'w') as stats_file:
        start_idx = 0
        if isinstance(stats_filecontent[0], str):
            stats_file.write(stats_filecontent[0])
            start_idx = 1
        csv_writer = csv.DictWriter(stats_file, stats_fields)
        csv_writer.writeheader()
        csv_writer.writerows(stats_filecontent[start_idx:])

