class StepFunction:
    def __init__(self):
        self.intervals = []
    
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

# 创建分段函数实例
sf = StepFunction()

# 添加初始区间
sf.add_interval(0, 4, 1) #4
sf.add_interval(4, 8, 2) #8
sf.add_interval(8, 12, 3) #12

# 测试积分
print(sf.integrate(2, 10))  # 输出: 16.0
print(sf.integrate(5, 15))  # 输出: 18.0
print(sf.integrate(10, 14)) # 输出: 6.0
print(sf.integrate(-2, 2))  # 输出: 2.0
print(sf.integrate(0, 0))   # 输出: 0.0
print(sf.integrate(0, 3.6))   # 输出: 3.6
print(sf.integrate(4.2, 8))   # 输出: 7.6

# 动态添加新区间
sf.add_interval(12, 16, 4)

# 重新测试积分
print(sf.integrate(5, 15))  # 输出: 30.0
print(sf.integrate(10, 14)) # 输出: 14.0
print(sf.integrate(14, 20)) # 输出: 8.0
print(sf.integrate(20, 22)) # 输出: 0.0