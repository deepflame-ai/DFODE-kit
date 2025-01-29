import os
os.system("python ./mp-random-each-sampling.py 1>>log.sampling")
os.system("python ./train.py 1>>log.training")
os.system("python ./aggregate_net.py 1>>log.aggregate_net")

