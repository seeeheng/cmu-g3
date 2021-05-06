from LocoBlock import LocoBlock
import numpy as np
import time

bot = LocoBlock()
# target_joint = [1.7050307268808753, 0.29153968814537734, 0.7706484461313616, 0.3661494329579619, 0.09594933475687878]
target_joint = [1.34622270699778, 0.31206635500859076, 0.8061984927842841, 0.3750729893573609, 0.10089961519631457]
bot.bot.arm.set_joint_positions(target_joint, plan=False)
print(bot.bot.arm.pose_ee)
time.sleep(1)