import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from datetime import datetime
from easydict import EasyDict as edict
from tqdm import tqdm
import logging
import json
import pickle
from collections import defaultdict
from config import cfg
import torch.distributed as dist


from torchlight.utils import initialize_exp, set_seed, get_dump_path


from src.data import load_data, Collator_base, EADataset
from src.utils import set_optim, Loss_log, pairwise_distances
from model import MEAformer
from src.distributed_utils import init_distributed_mode, reduce_value, cleanup
import torch.nn.functional as F
import gc
from model.layers import MetaModalityHybrid


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

writer = SummaryWriter(log_dir="logs")

def is_main_process():
    """
    Check if the current process is the main one in a distributed setup.
    """
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

class Runner:
    def __init__(self, args, writer=None, logger=None, rank=0):
        self.args = args
        self.writer = writer
        self.logger = logger or self.setup_logging()
        self.rank = rank
        self.model_list = []
        self.scaler = GradScaler()

        # Ensure log directory exists
        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)

    def setup_logging(self, log_dir="MEAformer-main/logs/"):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, "log.txt")
        logger = logging.getLogger("MEAformer")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def set_data_path(self):
        self.args.data_path = os.path.join(os.getcwd(), 'data')
        print(f"Data path set to: {self.args.data_path}")

    def run(self):
        self.loss_log = Loss_log()
        self.curr_loss = 0.0
        self.lr = self.args.lr
        self.step = 1
        self.epoch = 0

        with tqdm(total=self.args.epoch) as _tqdm:
            for i in range(self.args.epoch):
                if self.args.dist and not self.args.only_test:
                    self.train_sampler.set_epoch(i)

                self.epoch = i
                self.train(_tqdm)
                self.loss_log.update(self.curr_loss)
                self.update_loss_log()

                if (i + 1) % self.args.eval_epoch == 0:
                    self.eval()
                _tqdm.update(1)

    def train(self, _tqdm):
        self.model.train()
        curr_loss = 0.0
        accumulation_steps = self.args.accumulation_steps

        for batch in self.train_dataloader:
            loss, _ = self.model(
                batch['graph_feat'].to(self.args.device),
                batch['text_feat'].to(self.args.device),
                batch['image_feat'].to(self.args.device)
            )

            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()

            if self.args.dist:
                loss = reduce_value(loss, average=True)

            self.step += 1
            curr_loss += loss.item()

            if self.step % accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                for model in self.model_list:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                if not self.args.dist or is_main_process():
                    self.lr = self.scheduler.get_last_lr()[-1]
                    self.writer.add_scalars("lr", {"lr": self.lr}, self.step)

                for model in self.model_list:
                    model.zero_grad(set_to_none=True)

            if self.args.dist:
                torch.cuda.synchronize(self.args.device)

        return curr_loss


if __name__ == '__main__':
    cfgs = cfg()
    cfgs.get_args()
    cfgs.update_train_configs()

    # Set the random seed for reproducibility
    random_seed = cfgs.cfg.random_seed
    set_seed(random_seed)

    logger.info("Starting training...")

    # Close the writer and logger at the end
    if not cfgs.cfg.no_tensorboard and not cfgs.cfg.only_test:
        writer.close()
        logger.info("Training complete!")

    if cfgs.cfg.dist and not cfgs.cfg.only_test:
        dist.barrier()
        dist.destroy_process_group()


