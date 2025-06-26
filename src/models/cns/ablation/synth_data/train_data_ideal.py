import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'


import torch
from models.cns.sim.dataset import DataLoader, ObsAug
from models.cns.train_gvs_short_seq import train


class DataLoader_Ideal(DataLoader):
    def __init__(self, camera_config, batch_size, train=True, num_trajs=100, env="Point"):
        super().__init__(camera_config, batch_size, train, num_trajs, env)

        for dataset in self.datasets:
            dataset.aug = ObsAug.NoAug


if __name__ == "__main__":
    train(
        device=torch.device("cuda:0"),
        data_class=DataLoader_Ideal,
        suffix="data_ideal",
        save="posix" in os.name,
        # gui=True
    )
