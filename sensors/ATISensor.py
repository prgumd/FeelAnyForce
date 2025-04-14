import threading
import multiprocessing
import serial
import time
import asyncio
import numpy as np
import queue

class Receiver:
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyUSB0', 460800, timeout=1)
        self.sample_freq = 20
        self.sample_dt = 1 / self.sample_freq
        self.mode = "free"
        manager = multiprocessing.Manager()
        self.readings = manager.list()

    def send_char(self, char):
        self.ser.write(char.encode('utf-8'))
        self.ser.read()
        self.ser.write(b'\r')
        self.ser.read()
        self.ser.read()

    def send_str(self, str):
        for char in str:
            self.ser.write(char.encode('utf-8'))
            self.ser.read(1)

    def get_ft(self):
        if self.mode != "c":
            self.send_char('s')
            forces = self.ser.readline()
            forces = forces.split()
            return np.array([float(f.decode('utf-8')) for f in forces[::2]])
        print("print sensor is in c mode, call stop_ft first!")

    def _get_ft_async(self):
        self.mode = "c"
        self.send_char("c")
        self.ser.read()
        self.ser.read()
        while True:
            incoming_q = self.ser.in_waiting
            if incoming_q >= 1:
                try:
                    ft = self.ser.readline()
                    ft = ft.split()
                    ft = [float(f.decode('utf-8')) for f in ft[::2]]
                    self.readings.append(ft)
                except:
                    pass
            else:
                time.sleep(self.sample_dt / 2)

    def get_ft_async(self):  # have to call stop_ft to stop it
        if self.mode != "c":
            self.get_ft_thread = multiprocessing.Process(target=self._get_ft_async)
            self.get_ft_thread.start()

    def stop_ft(self):
        self.send_str("ccc\r")
        self.ser.read()
        self.ser.read()
        self.get_ft_thread.terminate()
        self.mode = "free"

    def get_mean_ft(self, period=5):
        self.get_ft_async()
        time.sleep(period)
        self.stop_ft()
        ft_readings = np.array(self.readings)
        return np.mean(ft_readings, axis=0)

    def tare(self):
        self.send_str("bias on\r")
        self.ser.readlines()

    def close(self):
        self.ser.close()


def main():
    sensor = Receiver()
    sensor.tare()
    print("sensor tared", sensor.get_ft())

if __name__ == "__main__":
    main()