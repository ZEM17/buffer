import numpy as np

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = './envivio/video_size_'

class StepFunction:
    def __init__(self, video_size):
        self.intervals = []
        self.video_size = video_size
        self.count = 0
    
    def add_interval(self, start, end, value):
        # 检查是否有重叠
        for interval in self.intervals:
            if not (end <= interval[0] or start >= interval[1]):
                raise ValueError("区间重叠")
        # 插入并保持有序
        import bisect
        bisect.insort(self.intervals, (start, end, value))

    def f(self, x):
        import bisect
        # 找到第一个 interval[0] > x
        idx = bisect.bisect_right(self.intervals, (x, float('inf'), float('inf')))
        if idx > 0:
            # 检查x是否在前一个区间内
            prev = self.intervals[idx-1]
            if prev[0] <= x < prev[1]:
                return prev[2]
        return 0  # 默认值
    
    def integrate(self, a, b):
        """
        计算在指定区间 [a, b) 内的积分。
        
        参数:
        a (float): 积分下限
        b (float): 积分上限
        
        返回:
        float: 积分结果
        """
        total = 0
        for interval in self.intervals:
            start, end, value = interval
            inter_start = max(a, start)
            inter_end = min(b, end)
            if inter_start < inter_end:
                total += value * (inter_end - inter_start)
        return total
    
    def add_chunk(self, bitrate):
        dense = self.video_size[bitrate][self.count] / 4
        self.add_interval(4 * self.count, 4 * self.count + 4, dense)
        self.count += 1


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

        self.progress = 0
        self.sf = StepFunction(self.video_size)

    def get_video_chunk(self, quality, max_buffer_size):

        assert quality >= 0
        assert quality < BITRATE_LEVELS
        assert max_buffer_size > 0
        max_buffer_size *= MILLISECONDS_IN_SECOND
        self.sf.add_chunk(quality)
        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        
        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                         * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                       - self.last_mahimahi_time
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

	    # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN
        self.progress += VIDEO_CHUNCK_LEN
        # print(self.progress - self.buffer_size, self.progress) # 区间

        # sleep if buffer > max_buffer
        sleep_time = 0
        if self.buffer_size > max_buffer_size:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - max_buffer_size
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                           - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0
        # sleep if buffer gets too large
        # sleep_time = 0
        # if self.buffer_size > BUFFER_THRESH:
        #     # exceed the buffer limit
        #     # we need to skip some network bandwidth here
        #     # but do not add up the delay
        #     drain_buffer_time = self.buffer_size - BUFFER_THRESH
        #     sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
        #                  DRAIN_BUFFER_SLEEP_TIME
        #     self.buffer_size -= sleep_time
        #
        #     while True:
        #         duration = self.cooked_time[self.mahimahi_ptr] \
        #                    - self.last_mahimahi_time
        #         if duration > sleep_time / MILLISECONDS_IN_SECOND:
        #             self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
        #             break
        #         sleep_time -= duration * MILLISECONDS_IN_SECOND
        #         self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
        #         self.mahimahi_ptr += 1
        #
        #         if self.mahimahi_ptr >= len(self.cooked_bw):
        #             # loop back in the beginning
        #             # note: trace file starts with time 0
        #             self.mahimahi_ptr = 1
        #             self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video

        buffer_data = self.sf.integrate((self.progress - self.buffer_size)/1000, (self.progress)/1000)
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            self.progress = 0
            self.sf = StepFunction(self.video_size)
            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            end_of_video, \
            video_chunk_remain, \
            buffer_data*1e-6
            