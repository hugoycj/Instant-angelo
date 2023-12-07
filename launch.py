import os
import torch
import logging
import datasets
import argparse
import systems
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from utils.callbacks import (
    CodeSnapshotCallback,
    ConfigSnapshotCallback,
    CustomProgressBar,
)
from datasets.colmap import ColmapDataModule
from utils.misc import load_config


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
    logger = logging.getLogger("pytorch_lightning")
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    pl.seed_everything(config.seed)
    system = systems.make(
        config.system.name,
        config,
        load_from_checkpoint=None if not args.resume_weights_only else args.resume,
    )
    dm = ColmapDataModule(config.dataset, torch.device("cuda:0"))
    callbacks = []
    if args.train:
        callbacks += [
            ModelCheckpoint(dirpath=config.ckpt_dir, **config.checkpoint),
            LearningRateMonitor(logging_interval="step"),
            CodeSnapshotCallback(config.code_dir, use_version=False),
            ConfigSnapshotCallback(config, config.config_dir, use_version=False),
            CustomProgressBar(refresh_rate=1),
        ]

    loggers = []
    if args.train:
        loggers += [
            TensorBoardLogger(
                args.runs_dir, name=config.name, version=config.trial_name
            ),
            CSVLogger(config.exp_dir, name=config.trial_name, version="csv_logs"),
        ]

    trainer = Trainer(
        devices=args.gpu,
        accelerator="gpu",
        callbacks=callbacks,
        logger=loggers,
        strategy="ddp_find_unused_parameters_false",
        **config.trainer,
    )

    if args.train:
        if args.resume and not args.resume_weights_only:
            trainer.fit(system, datamodule=dm, ckpt_path=args.resume)
        else:
            trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm, ckpt_path=args.resume)
    elif args.test:
        trainer.test(system, datamodule=dm, ckpt_path=args.resume)
    elif args.predict:
        trainer.predict(system, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
