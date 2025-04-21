"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR or DINO:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
https://github.com/facebookresearch/dino/blob/main/utils.py
"""
import argparse
import time
import datetime
from collections import defaultdict, deque
import numpy as np
import torch
from tqdm import tqdm


def find_max_and_min(arr):
    """
    Finds the min and max values in a list of depth image file paths.
    """
    first_image = np.load(arr[0])
    max_val = first_image.max()
    min_val = first_image.min()

    for i in tqdm(arr[1:], desc="Computing min/max"):
        im = np.load(i)
        max_val = max(max_val, im.max())
        min_val = min(min_val, im.min())

    return max_val, min_val


def compute_mean_std(loader):
    """
    Computes the mean and standard deviation for the dataset.
    """
    sum_rgb = torch.zeros(3)
    sum_rgb_sq = torch.zeros(3)
    sum_depth = torch.zeros(3)
    sum_depth_sq = torch.zeros(3)
    nb_samples = 0

    for rgb, depth in loader:
        batch_samples = rgb.size(0)
        nb_samples += batch_samples

        rgb = rgb.view(batch_samples, rgb.size(1), -1)  
        depth = depth.view(batch_samples, depth.size(1), -1)  

        sum_rgb += rgb.mean(dim=2).sum(dim=0)
        sum_depth += depth.mean(dim=2).sum(dim=0)

    mean_rgb = sum_rgb / nb_samples
    mean_depth = sum_depth / nb_samples

    for rgb, depth in loader:
        batch_samples = rgb.size(0)

        rgb = rgb.view(batch_samples, rgb.size(1), -1)
        depth = depth.view(batch_samples, depth.size(1), -1)

        sum_rgb_sq += ((rgb - mean_rgb.view(1, 3, 1)) ** 2).mean(dim=2).sum(dim=0)
        sum_depth_sq += ((depth - mean_depth.view(1, 3, 1)) ** 2).mean(dim=2).sum(dim=0)

    std_rgb = torch.sqrt(sum_rgb_sq / nb_samples)
    std_depth = torch.sqrt(sum_depth_sq / nb_samples)

    return mean_rgb.tolist(), std_rgb.tolist(), mean_depth.tolist(), std_depth.tolist()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))



