import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from share import *
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, compare_weights

pl.seed_everything(42, workers=True) #固定所有隨機數產生器（Python, Numpy, PyTorch）
torch.backends.cudnn.benchmark = False #關閉自動尋找最快算法的功能（因為最快算法通常不確定）
torch.backends.cudnn.deterministic = True # 強制要求使用確定性的算法

# Configs
resume_path = '/media/Siamese-Diffusion/models/control_sd15_ini.ckpt' #controlnet權重
batch_size = 6
logger_freq = 400
learning_rate = 1.96e-05
sd_locked = False #開啟梯度更新
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('/media/Siamese-Diffusion/models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False) #允許權重檔案與模型結構部分吻合

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=2, batch_size=batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq)
custom_log = TensorBoardLogger(save_dir="my_project_logs_v129", name="experiment_v130")
trainer = pl.Trainer(strategy="auto", accelerator="gpu", devices=1, callbacks=[logger], logger=custom_log,deterministic=True, max_steps=5000)
# Train!
trainer.fit(model, dataloader)
