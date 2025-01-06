import multiprocessing as mp
import numpy as np
import os
from env import ABREnv
import ppo2actor_cuda as network
import torch
import pandas as pd

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [7, 8]
A_DIM = 6
A2_DIM = 3
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 2
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 500000
MODEL_SAVE_INTERVAL = 500
RANDOM_SEED = 42
SUMMARY_DIR = './ppo_2actor_buffer'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    # os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python test_HA.py ' + nn_model)

    # append test performance to the log
    rewards, entropies, buffers, max_buffers, buffer_occupancys = [], [], [], [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy, buffer, max_buffer, buffer_occupancy = [], [], [], [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    buffer_occupancy.append(float(parse[-3]))
                    max_buffer.append(float(parse[-4]))
                    buffer.append(float(parse[2]))
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))
        buffers.append(np.mean(buffer[1:]))
        max_buffers.append(np.mean(max_buffer[1:]))
        buffer_occupancys.append(np.mean(buffer_occupancy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies), np.mean(buffers), np.mean(max_buffers), np.mean(buffer_occupancys)
        
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM, 
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        summary_reward = {
            'ep': [],
            'reward': [],
            'entropy': [],
            'entropy_weight': [],
            'buffer': [],
            'max_buffer': [],
        }
        summary_loss = {
            'ep': [],
            'actor_loss': [],
            'critic_loss': [],
        }
        pd.DataFrame(summary_reward).to_csv(SUMMARY_DIR + '/summary_reward.csv', index=False)
        pd.DataFrame(summary_loss).to_csv(SUMMARY_DIR + '/summary_loss.csv', index=False)

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            actor.load_model(nn_model)
            print('Model restored.')
        
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a1, a2, p1, p2, r, adv = [], [], [], [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a1_, a2_, p1_, p2_, r_, adv_ = exp_queues[i].get()
                s += s_
                a1 += a1_
                a2 += a2_
                p1 += p1_
                p2 += p2_
                r += r_
                adv += adv_
            s_batch = np.stack(s, axis=0)
            a1_batch = np.vstack(a1)
            a2_batch = np.vstack(a2)
            p1_batch = np.vstack(p1)
            p2_batch = np.vstack(p2)
            v_batch = np.vstack(r)
            adv_batch = np.vstack(adv)
            actor.train(s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, adv_batch, epoch)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
                
                avg_reward, avg_entropy, avg_buffer, avg_maxbuf, avg_buff_occupy= testing(epoch,
                    SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth', 
                    test_log_file)

                print("epoch:{}, reward:{}, buffer:{}, maxbuf:{}".format(epoch, avg_reward, avg_buffer, avg_maxbuf))

                summary_reward = {
                    'ep': [epoch],
                    'reward': [avg_reward],
                    'entropy': [avg_entropy],
                    'entropy_weight': [actor._entropy_weight],
                    'buffer': [avg_buffer],
                    'max_buffer': [avg_maxbuf]
                }
                pd.DataFrame(summary_reward).to_csv(SUMMARY_DIR + '/summary_reward.csv', mode='a', index=False, header=False)


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a1_batch, a2_batch, p1_batch, p2_batch, r_batch = [], [], [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action1_prob, action2_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            # gumbel noise
            noise = np.random.gumbel(size=len(action1_prob))
            bit_rate = np.argmax(np.log(action1_prob) + noise)
            max_buffer_opt = np.random.choice(len(action2_prob), size=1, p=action2_prob)[0]

            if max_buffer_opt == 0 and env.max_buffer_size > 5:
                env.max_buffer_size -= 5
            elif max_buffer_opt == 1:
                env.max_buffer_size += 0
            elif max_buffer_opt == 2 and env.max_buffer_size < 55:
                env.max_buffer_size += 5
            elif max_buffer_opt == 2 and env.max_buffer_size > 55:
                env.max_buffer_size = 60

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a1_batch.append(action_vec)

            action_vec = np.zeros(A2_DIM)
            action_vec[max_buffer_opt] = 1
            a2_batch.append(action_vec)

            r_batch.append(rew)
            p1_batch.append(action1_prob)
            p2_batch.append(action2_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, r_batch, done)
        gae_batch = actor.compute_gae(r_batch, s_batch)
        exp_queue.put([s_batch, a1_batch, a2_batch, p1_batch, p2_batch, v_batch, gae_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():
    mp.set_start_method('spawn')
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
