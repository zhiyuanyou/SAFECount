import argparse
import logging
import os
import pprint
import shutil
import time

import torch
import torch.distributed as dist
import yaml
from datasets.data_builder import build_dataloader
from easydict import EasyDict
from models.model_helper import build_network
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.criterion_helper import build_criterion
from utils.dist_helper import setup_distributed
from utils.eval_helper import dump, merge_together, performances
from utils.lr_helper import get_scheduler
from utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    to_device,
)
from utils.optimizer_helper import get_optimizer
from utils.vis_helper import build_visualizer

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument(
    "-c", "--config", type=str, default="./config.yaml", help="Path of config"
)
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("--local_rank", default=None, help="local rank for dist")


def main():
    global args, config, best_mae, best_rmse, visualizer, lr_scale_backbone
    args = parser.parse_args()

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    config.exp_path = os.path.dirname(args.config)
    config.save_path = os.path.join(config.exp_path, config.saver.save_dir)
    config.log_path = os.path.join(config.exp_path, config.saver.log_dir)
    config.evaluator.eval_dir = os.path.join(config.exp_path, config.evaluator.save_dir)
    if (args.evaluate or args.test) and config.get("visualizer", None):
        config.visualizer.vis_dir = os.path.join(
            config.exp_path, config.visualizer.vis_dir
        )
        visualizer = build_visualizer(**config.visualizer)

    config.port = config.get("port", None)
    rank, world_size = setup_distributed(port=config.port)

    if rank == 0:
        os.makedirs(config.save_path, exist_ok=True)
        os.makedirs(config.log_path, exist_ok=True)
        if (args.evaluate or args.test) and config.get("visualizer", None):
            os.makedirs(config.visualizer.vis_dir, exist_ok=True)

        current_time = get_current_time()
        logger = create_logger(
            "global_logger", config.log_path + "/dec_{}.log".format(current_time)
        )
        logger.info("\nargs: {}".format(pprint.pformat(args)))
        logger.info("\nconfig: {}".format(pprint.pformat(config)))

    random_seed = config.get("random_seed", None)
    reproduce = config.get("reproduce", None)
    if random_seed:
        set_random_seed(random_seed, reproduce)

    criterion = build_criterion(config.criterion)

    # create model
    model = build_network(config.net)
    model.cuda()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # parameters
    model.train()
    lr_scale_backbone = config.trainer["lr_scale_backbone"]
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False
        # parameters not include backbone
        parameters = [
            p for n, p in model.module.named_parameters() if "backbone" not in n
        ]
    else:
        assert lr_scale_backbone > 0 and lr_scale_backbone <= 1
        parameters = [
            {
                "params": [
                    p
                    for n, p in model.module.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ],
                "lr": config.trainer.optimizer.kwargs.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.module.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": lr_scale_backbone * config.trainer.optimizer.kwargs.lr,
            },
        ]

    optimizer = get_optimizer(parameters, config.trainer.optimizer)
    lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
    last_epoch = 0
    best_mae = float("inf")
    best_rmse = float("inf")

    # load model: auto_resume > resume_model > load_path
    auto_resume = config.saver.get("auto_resume", True)
    resume_model = config.saver.get("resume_model", None)
    load_path = config.saver.get("load_path", None)

    if resume_model and not resume_model.startswith("/"):
        resume_model = os.path.join(config.exp_path, resume_model)
    lastest_model = os.path.join(config.save_path, "ckpt.pth.tar")
    if auto_resume and os.path.exists(lastest_model):
        resume_model = lastest_model
    if resume_model:
        best_mae, last_epoch = load_state(resume_model, model, optimizer=optimizer)
    elif load_path:
        if not load_path.startswith("/"):
            load_path = os.path.join(config.exp_path, load_path)
        load_state(load_path, model)

    train_loader, val_loader, test_loader = build_dataloader(
        config.dataset, distributed=True
    )

    if args.evaluate:
        val_mae, val_rmse = eval(val_loader, model, criterion)
        return

    if args.test:
        test_mae, test_rmse = eval(test_loader, model, criterion)
        return

    for epoch in range(last_epoch, config.trainer.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch)
        lr_scheduler.step(epoch + 1)

        val_mae, val_rmse = eval(val_loader, model, criterion)

        if rank == 0:
            is_best = False
            if best_mae >= val_mae:
                is_best = True
                best_mae = val_mae
                best_rmse = val_rmse
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_metric": best_mae,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                config,
            )


def train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch):

    model.train()
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False

    logger = logging.getLogger("global_logger")
    rank = dist.get_rank()
    if rank == 0:
        logger.info("Training on train set dataset")
    train_loss = 0

    end = time.time()
    for i, sample in enumerate(train_loader):
        time_data = time.time() - end

        iter = i + 1
        current_lr = lr_scheduler.get_lr()[0]
        sample = to_device(sample, device=torch.device("cuda"))
        # forward
        outputs = model(sample)  # 1 x 1 x h x w
        loss = 0
        for name, criterion_loss in criterion.items():
            weight = criterion_loss.weight
            loss += weight * criterion_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        density = outputs["density"]
        density_pred = outputs["density_pred"]
        train_loss += loss.item()
        pred_cnt = torch.sum(density_pred).item()
        gt_cnt = torch.sum(density).item()

        time_epoch = time.time() - end
        end = time.time()

        if rank == 0:
            logger.info(
                "Train | Epoch : {} / {} | Iter: {} / {} | lr: {} | Data: {:.2f}, Time: {:.2f} | Loss: {}".format(
                    epoch + 1,
                    config.trainer.epochs,
                    iter,
                    len(train_loader),
                    current_lr,
                    time_data,
                    time_epoch,
                    loss,
                )
            )
            logger.info(
                "Train | GT: {:5.1f}, Pred: {:5.1f} | Best Val MAE: {}, Best Val RMSE: {}".format(
                    gt_cnt, pred_cnt, best_mae, best_rmse
                )
            )

    if rank == 0:
        logger.info("gather final results")
    train_loss = torch.Tensor([train_loss]).cuda()
    iter = torch.Tensor([iter]).cuda()
    dist.all_reduce(train_loss)
    dist.all_reduce(iter)
    train_loss = train_loss.item() / iter.item()

    if rank == 0:
        logger.info(
            "Finish Train Epoch: {} | Average Loss: {} | Best Val MAE: {}, Best Val RMSE: {}".format(
                epoch + 1, train_loss, best_mae, best_rmse
            )
        )


def eval(val_loader, model, criterion):
    model.eval()
    logger = logging.getLogger("global_logger")
    rank = dist.get_rank()
    if rank == 0:
        logger.info("Evaluation on val dataset or test dataset")

    if rank == 0:
        os.makedirs(config.evaluator.eval_dir, exist_ok=True)
    # all threads write to config.evaluator.eval_dir, it must be made before every thread begin to write
    dist.barrier()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            iter = i + 1
            sample = to_device(sample, device=torch.device("cuda"))
            outputs = model(sample)
            loss = 0
            for name, criterion_loss in criterion.items():
                weight = criterion_loss.weight
                loss += weight * criterion_loss(outputs)

            dump(config.evaluator.eval_dir, outputs)

            density = outputs["density"]
            density_pred = outputs["density_pred"]
            pred_cnt = torch.sum(density_pred).item()
            gt_cnt = torch.sum(density).item()

            if (args.evaluate or args.test) and config.get("visualizer", None):
                visualizer.vis_batch(outputs)
            logger.info(outputs["filename"])
            if rank == 0:
                logger.info(
                    "Val | Iter: {} / {} | GT: {:5.1f}, Pred: {:5.1f} | Best Val MAE: {}, Best Val RMSE: {}".format(
                        iter, len(val_loader), gt_cnt, pred_cnt, best_mae, best_rmse
                    )
                )

    # gather final results
    dist.barrier()
    if rank == 0:
        logger.info("gather final results")

    val_mae = None
    val_rmse = None
    if rank == 0:
        gt_cnts, pred_cnts = merge_together(config.evaluator.eval_dir)
        val_mae, val_rmse = performances(gt_cnts, pred_cnts)
        shutil.rmtree(config.evaluator.eval_dir)
        logger.info(
            "Finish Val | MAE: {:5.2f}, RMSE: {:5.2f} | Best Val MAE: {}, Best Val RMSE: {}".format(
                val_mae, val_rmse, best_mae, best_rmse
            )
        )

    model.train()
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False

    return val_mae, val_rmse


if __name__ == "__main__":
    main()
