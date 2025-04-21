import numpy as np
import time
from sensors import ATISensor
from datetime import datetime
import csv
import pandas as pd
from sensors.gsrobotics import gsdevice, gs3drecon
import os
import cv2
import glob
import ast
import time
import yaml
import math as m
import click

np.seterr(all="ignore")

np.set_printoptions(precision=6)


class colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'


def init_mini():
    dev = gsdevice.Camera("GelSight Mini")
    net_file_path = 'nnmini.pt'
    GPU = False
    dev.connect()

    ''' Load neural network '''
    net_path = os.path.join("sensors", "gsrobotics", net_file_path)
    print('mini net path = ', net_path)

    if GPU:
        gpuorcpu = "cuda"
    else:
        gpuorcpu = "cpu"

    nn = gs3drecon.Reconstruction3D(dev)
    net = nn.load_nn(net_path, gpuorcpu)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    return dev, nn


def get_tacdepth(dev, nn):
    for _ in range(10):
        f1 = dev.get_image()
        dm = nn.get_depthmap(f1, False)
    return f1, dm


def update_record(base_dir, record_df, index, FT, tactile, depth_map):
    current_time_millis = int(round(time.time() * 1000))
    tactile_dir = os.path.join(base_dir, "tactile", f"{current_time_millis}.png")
    depth_dir = os.path.join(base_dir, "depth", f"{current_time_millis}.npy")
    cv2.imwrite(tactile_dir, tactile)
    np.save(depth_dir, depth_map)
    ft_str = ' '.join(f'{x:.6f}' for x in FT)
    record_df.loc[index] = [tactile_dir, depth_dir, ft_str]


@click.command()
@click.option('-c', '--record_count', type=int, default=1000)
@click.option('-i', '--interval', type=int, default=1)
def main(record_count, interval):
    indenter_name = "calibration_sphere"

    ft_r = ATISensor.Receiver()
    ft_r.tare()

    dev, nn = init_mini()
    for _ in range(10): get_tacdepth(dev, nn)

    records_df = pd.DataFrame(index=range(record_count), columns=["tactile", "depth", "FT"])

    exp_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("dataset", indenter_name)
    os.makedirs(os.path.join(base_dir, "tactile"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "depth"), exist_ok=True)
    tactile_dir_bg = os.path.join(base_dir, "background.png")
    tactile, dm = get_tacdepth(dev, nn)
    cv2.imwrite(tactile_dir_bg, tactile)

    # Get the latest CSV file in the directory
    csvs = glob.glob(os.path.join(base_dir, '*.csv'))
    if csvs:
        latest_csv = max(csvs, key=os.path.getctime)
    else:
        latest_csv = os.path.join(base_dir, f"calibration.csv")
    records_df.to_csv(latest_csv, index=False, mode='w', header=True)

    records_df = pd.read_csv(latest_csv)
    index = 0
    while index < len(records_df):
        row = records_df.iloc[index]
        tactile, dm = get_tacdepth(dev, nn)
        cv2.imwrite("last_tac.png", tactile)
        update_record(base_dir, records_df, index, ft_r.get_ft(), tactile, dm)
        print(colors.BLUE, 20 * "_", f"step: {index}/{len(records_df)}", 20 * "_",
              colors.END)
        print("FT", records_df["FT"].iloc[index])
        time.sleep(interval)

        records_df.to_csv(latest_csv, index=False, mode='w', header=True)
        index += 1


if __name__ == "__main__":
    main()
