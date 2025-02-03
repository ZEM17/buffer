from env import ABREnv

env = ABREnv(1)
for i in range(30):
    action1, action2 = env.hibrid_action(i)
    print(action1, action2)