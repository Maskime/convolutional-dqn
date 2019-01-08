import logging

# création de l'objet logger qui va nous servir à écrire dans les logs
logger = logging.getLogger()
# on met le niveau du logger à DEBUG, comme ça il écrit tout
logger.setLevel(logging.DEBUG)

# création d'un formateur qui va ajouter le temps, le niveau
# de chaque message quand on écrira un message dans le log
cdqn_logformat = '%(asctime)s :: %(levelname)s :: %(message)s'
formatter = logging.Formatter(cdqn_logformat)
# création d'un handler qui va rediriger une écriture du log vers
# un fichier en mode 'append', avec 1 backup et une taille max de 1Mo
# file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
# # on lui met le niveau sur DEBUG, on lui dit qu'il doit utiliser le formateur
# # créé précédement et on ajoute ce handler au logger
# file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# création d'un second handler qui va rediriger chaque écriture de log
# sur la console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)