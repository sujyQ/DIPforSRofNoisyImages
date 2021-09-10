import os
os.environ["CUDA_VISIBLE_DEVICES"]="2, 3"
from solver import Solver
import time
import numpy as np

from configs import getConfigs
cfg = getConfigs()

Set5_list = ["baby", "bird", "butterfly", "head", "woman"]
# Set14_list = ['baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers', 'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra']

start_time = time.time()
for img in Set5_list :
    print(img)
    trainer = Solver(cfg, img)
    trainer.train()
    trainer.finalize()
print("total : ", (time.time() - start_time)/60, "(minutes)")