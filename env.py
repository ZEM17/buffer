# add queuing delay into halo
import os
import numpy as np
import core as abrenv
import load_trace

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 7
S_LEN = 8  # take how many frames in the past K
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
BIT_RATE_PENALTY = np.array([1.02666667, 1.06666667, 1.10666667, 1.16444444, 1.25333333, 1.38222222])
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6
LAMDA = 0.9

class ABREnv():

    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                          all_cooked_bw=all_cooked_bw,
                                          random_seed=random_seed)

        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self.max_buffer_size = 60
        self.buffer_occupancy = 0.
        self.buffer_weight = 0.2

    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        bit_rate = self.last_bit_rate
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate, self.max_buffer_size)
        self.buffer_occupancy = self.buffer_size / self.max_buffer_size
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        # q_t
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        # b_t
        state[1, -1] = self.buffer_occupancy  # 10 sec
        # c_t
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        # state[2, :] = float(video_chunk_size) / \
        #     float(delay) / M_IN_K  # kilo byte / ms
        # m_t
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        # # no use
        # state[4, :A_DIM] = np.array(
        #     next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        # # no use
        # state[5, -1] = np.minimum(video_chunk_remain,
        #                           CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        avg_video_chunk_sizes = np.zeros(A_DIM)
        for i in range(A_DIM):
            avg_video_chunk_sizes[i] = np.mean(self.net_env.video_size[i])
        state[4, :A_DIM] = avg_video_chunk_sizes / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = self.max_buffer_size
        state[6, -1] = self.buffer_weight
        self.state = state
        return state

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        last_buffer_size = self.buffer_size

        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate, self.max_buffer_size)
        self.buffer_occupancy = self.buffer_size / self.max_buffer_size

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms




        # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
        #     - REBUF_PENALTY * rebuf \
        #     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
        #                               VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K \
        #     - self.buffer_weight * self.buffer_size / 10\
            # - self.buffer_weight * self.buffer_size * (bit_rate+1)

        reward = LAMDA * ( VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K ) \
            + (1 - LAMDA) * (-1 * self.buffer_size / BUFFER_NORM_FACTOR)

        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        # state 4 不用动
        state[5, -1] = self.max_buffer_size
        state[6, -1] = self.buffer_weight

        self.state = state
        #observation, reward, done, info = env.step(action)
        return state, reward, end_of_video, {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf}
