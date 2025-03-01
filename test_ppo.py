import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
from algorithm.ppo2_addsomething import Network
import fixed_env as env

S_INFO = 7  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
S_DIM = [7, 8]
A_DIM = (6, 5)
A1_DIM = 6
LR = 1e-4
FEATURE_NUM = 64
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
BUFFER_WEIGH = 0.1
LOG_FILE = './test_results/log_sim_ppo'
TEST_PATH = './test_results/'

TEST_TRACES = './test/'
SAVE_PATH = "./ppo_model/"

if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

def hibrid_action(action):
    if not (0 <= action < 30):
        raise ValueError("action3_num must be between 0 and 29 inclusive.")
    action1 = action % 6       # 计算 action1：取模运算
    action2 = action // 6      # 计算 action2：整除运算
    return action1, action2

def test(NN_MODEL):

    np.random.seed(RANDOM_SEED)


    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)
    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    agent = Network(S_DIM, 6, LR, FEATURE_NUM)
    if NN_MODEL is not None: 
        agent.load_model(SAVE_PATH+"nn_model_"+str(NN_MODEL)+".pth")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY


    s_batch = [np.zeros((S_INFO, S_LEN))]
    r_batch = []
    buffer_batch = []
    data_size_batch = []
    video_count = 0

    buffer_weight = BUFFER_WEIGH
    max_buffer_size = 30
    
    reward_per_trace = []
    buffer_per_trace = []
    data_size_per_trace = []
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, throughput_MB, data_size= \
            net_env.get_video_chunk(bit_rate, max_buffer_size)
        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smoothness
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                    - REBUF_PENALTY * rebuf \
                    - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                            VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)
        buffer_batch.append(buffer_size)
        data_size_batch.append(data_size)
        last_bit_rate = bit_rate

        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(data_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(max_buffer_size) + '\t' +
                        str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        avg_video_chunk_sizes = np.zeros(A1_DIM)
        for i in range(A1_DIM):
            avg_video_chunk_sizes[i] = np.mean(net_env.video_size[i])
        state[4, :A1_DIM] = avg_video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = max_buffer_size
        state[6, -1] = buffer_weight

        action1_prob, action2_prob = agent.predict(
            np.reshape(state, (1, S_DIM[0], S_DIM[1])))

        # gumbel noise
        noise = np.random.gumbel(size=len(action1_prob))
        bit_rate = np.argmax(np.log(action1_prob) + noise)
        max_buffer_opt = np.random.choice(len(action2_prob), size=1, p=action2_prob)[0]
        # bit_rate, max_buffer_opt = hibrid_action(action)
        # noise = np.random.gumbel(size=len(action1_prob))
        # bit_rate = np.argmax(np.log(action1_prob) + noise)
        # max_buffer_opt = np.random.choice(len(action2_prob), size=1, p=action2_prob)[0]

        if max_buffer_opt == 0:
            max_buffer_size = max(max_buffer_size-10,4)
        elif max_buffer_opt == 1:
            max_buffer_size = max(max_buffer_size-5,4)
        elif max_buffer_opt == 2:
            max_buffer_size = max_buffer_size
        elif max_buffer_opt == 3:
            max_buffer_size = min(max_buffer_size+5,60)
        elif max_buffer_opt == 4:
            max_buffer_size = min(max_buffer_size+10,60)

        s_batch.append(state)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            max_buffer_size = 30

            reward_per_trace.append(np.mean(r_batch[1:]))
            buffer_per_trace.append(np.mean(buffer_batch[1:]))
            data_size_per_trace.append(np.mean(data_size_batch[1:]))
            # print(np.mean(r_batch[1:]))
            del s_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A1_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            # print(np.mean(entropy_record))

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

    # print("step:", NN_MODEL, "avg_reward:",np.mean(reward_per_trace))
    return np.mean(reward_per_trace), np.mean(buffer_per_trace), np.mean(data_size_per_trace)