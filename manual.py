from Locobot import Locobot
import numpy as np
import time

bot = Locobot()
# target_joint = [1.7050307268808753, 0.29153968814537734, 0.7706484461313616, 0.3661494329579619, 0.09594933475687878]
target_joint = [1.6708857219702318, 0.3689038829848021, 0.8707715680214299, 0.4063690073562926, 0.09972229203668327]
bot.bot.arm.set_joint_positions(target_joint, plan=False)
print(bot.bot.arm.pose_ee)
time.sleep(1)