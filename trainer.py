import os
import torch
import datasets
import argparse
import systems
from pydlutils.torch import seed
from loguru import logger
from datetime import datetime
from utils.misc import load_config
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, config):
        self.cfg = config.trainer
        self.config = config
        self.device = torch.device("cuda")
        self.writer = SummaryWriter(f"{config.exp_dir}/{config.trial_name}")
        self.global_step = 0

    def test(self, system, datamodule):
        logger.info("Testing .....")
        dataloader = datamodule.test_dataloader()
        system.model.eval()
        for bidx, batch in enumerate(dataloader):
            system.on_test_batch_start(batch, dataloader.dataset)
            system.test_step(batch, bidx)
        system.model.train()

    def train(self, system, datamodule, ckpt_path):
        max_epoch = self.cfg.get("max_epoch", 1)
        cfg = self.cfg
        system.setup(self.writer, self.device)
        datamodule.setup(stage=None, device=self.device)
        optimizer, scheduler = system.configure_optimizers()
        for epoch in range(max_epoch):
            dataloader = datamodule.train_dataloader()
            system.model.train()
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                system.update_status(epoch, self.global_step)
                system.on_train_batch_start(batch, dataloader.dataset)
                loss = system.training_step(batch, batch_idx)
                loss["loss"].backward()
                optimizer.step()
                scheduler.step()
                self.global_step += 1
                if self.global_step % cfg.log_every_n_steps == 0 or batch_idx == 0:
                    logger.info(f"Epoch {epoch}: {self.global_step}/{cfg.max_steps}")

                if self.global_step % cfg.val_check_interval == 0:
                    self.test(system, datamodule)

                if self.global_step == cfg.max_steps:
                    break

        system.export()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument(
        "--resume", default=None, help="path to the weights to be resumed"
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="specify this argument to restore only the weights (w/o training states), e.g. --resume path/to/resume --resume_weights_only",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")
    group.add_argument("--predict", action="store_true")
    # group.add_argument('--export', action='store_true') # TODO: a separate export action

    parser.add_argument("--exp_dir", default="./exp")
    parser.add_argument("--runs_dir", default="./runs")
    parser.add_argument(
        "--verbose", action="store_true", help="if true, set logging level to DEBUG"
    )

    args, extras = parser.parse_known_args()
    # parse YAML config to OmegaConf
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)
    config.trial_name = f"{config.tag}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config.exp_dir = f"{args.exp_dir}/{config.name}"
    config.save_dir = f"{config.exp_dir}/{config.trial_name}/save"
    config.ckpt_dir = os.path.join(config.exp_dir, config.trial_name, "ckpt")
    config.code_dir = os.path.join(config.exp_dir, config.trial_name, "code")
    config.config_dir = os.path.join(config.exp_dir, config.trial_name, "config")
    seed.set_seed(config.seed)
    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(
        config.system.name,
        config,
        load_from_checkpoint=None if not args.resume_weights_only else args.resume,
    )
    trainer = Trainer(config)
    trainer.train(system, dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
