from __future__ import print_function
import airsim
import cv2, sys, os
from f110_gym.sim_f110_core import SIM_f110Env
import numpy as np
import pickle

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def main():
    env = SIM_f110Env()
    obs = env.reset()
    lidarlist = []
    while True:
        #display lidar
        lidar = obs["lidar"]

        lidarlist.append(lidar)

        action = {"angle":0.0, "speed":1.0}
        obs, _, done, _ = env.step(action)
        
        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        if done:
            print("ISDONE")
            break
            # obs = env.reset()  

    with open("lidarlist.pkl", "wb") as f:
        pickle.dump(lidarlist, f)

if __name__ == '__main__':
    main()