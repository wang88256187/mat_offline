import torch
import numpy as np
from torch.utils.data import Dataset
import pickle
from mat.algorithms.utils.util import check


def parse_data(data):
    obs = []
    act = []
    ava = []
    for episode in data:
        num_agent = len(episode)
        episode_length = len(episode[0])
        for step in range(episode_length):
            step_obs, step_act, step_ava = [], [], []
            for agent in range(num_agent):
                # (s, o, a, r, d, ava)
                step_obs.append(episode[agent][step][1])
                step_act.append(episode[agent][step][2])
                step_ava.append(episode[agent][step][5])
            obs.append(step_obs)
            act.append(step_act)
            ava.append(step_ava)
    return np.array(obs, dtype=np.float32), np.array(act, dtype=np.float32), np.array(ava, dtype=np.float32)


class OfflineBuffer(Dataset):

    def __init__(self, data_dir):
        print("offline_data_dir: ", data_dir)
        with open(data_dir, "rb") as d:
            data = pickle.load(d)
        self.obs, self.act, self.ava = parse_data(data)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.ava[idx]
