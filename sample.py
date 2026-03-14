import sys
import torch
import os
import argparse
import logging
import time
import json
from pathlib import Path
from datetime import datetime

from utils.collate_utils import collate
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args
from utils.train_utils import set_random_seed
from utils import init_env
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(7)


def setup_logging(results_folder):
    log_dir = Path(results_folder) / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger


def get_loader(cfg, target_datasets, logger):
    """按需加载测试数据集，跳过路径不存在或配置缺失的数据集。"""
    loaders = {}
    for name in target_datasets:
        try:
            dataset = instantiate_from_config(getattr(cfg.test_dataset, name))
            loaders[name] = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate)
            logger.info(f"Loaded {name}: {len(dataset)} samples")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            loaders[name] = None

    failed = [k for k, v in loaders.items() if v is None]
    if failed:
        logger.warning(f"Skipped datasets: {failed}")
    return loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_sample_steps', type=int, default=None)
    parser.add_argument('--target_dataset', nargs='+', type=str,
                        default=['USOD10K', 'UFO120', 'SUIM', 'USOD'])
    parser.add_argument('--time_ensemble', action='store_true')
    parser.add_argument('--batch_ensemble', action='store_true')

    cfg = add_args(parser)
    logger = setup_logging(cfg.results_folder)

    assert not (cfg.time_ensemble and cfg.batch_ensemble), \
        'time_ensemble and batch_ensemble are mutually exclusive'

    if cfg.num_sample_steps is not None:
        cfg.diffusion_model.params.num_sample_steps = cfg.num_sample_steps

    # 加载数据集
    loaders = get_loader(cfg, cfg.target_dataset, logger)
    available = [k for k, v in loaders.items() if v is not None]
    missing = [d for d in cfg.target_dataset if loaders.get(d) is None]
    assert not missing, f"Requested datasets failed to load: {missing}"

    # 初始化模型
    cond_uvit = instantiate_from_config(
        cfg.cond_uvit,
        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass)
    )
    model = recurse_instantiate_from_config(cfg.model, unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model, model=model)
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())

    trainer = Trainer(
        diffusion_model,
        train_loader=None,
        test_loader=None,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=cfg.results_folder,
        optimizer=optimizer,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )
    trainer.load(pretrained_path=cfg.checkpoint)

    # 用accelerator prepare所有loader
    valid_loaders = [loaders[k] for k in available]
    prepared = trainer.accelerator.prepare(*valid_loaders)
    if len(valid_loaders) == 1:
        prepared = [prepared]
    loaders = {k: prepared[i] for i, k in enumerate(available)}

    # 逐数据集评估
    all_results = {}
    total_start = time.time()

    for dataset_name in cfg.target_dataset:
        loader = loaders[dataset_name]
        dataset_cfg = getattr(cfg.test_dataset, dataset_name)
        save_to = Path(cfg.results_folder) / dataset_name
        save_to.mkdir(exist_ok=True)

        gt_root = dataset_cfg.params.gt_root if hasattr(dataset_cfg.params, 'gt_root') \
            else str(Path(dataset_cfg.params.image_root).parent.parent)
        mask_path = Path(gt_root)

        logger.info(f"Evaluating {dataset_name}...")
        trainer.model.eval()
        t0 = time.time()

        if cfg.batch_ensemble:
            mae, _ = trainer.val_batch_ensemble(
                model=trainer.model, test_data_loader=loader,
                accelerator=trainer.accelerator, thresholding=True, save_to=save_to)
        elif cfg.time_ensemble:
            mae, _ = trainer.val_time_ensemble(
                model=trainer.model, test_data_loader=loader,
                accelerator=trainer.accelerator, thresholding=True, save_to=save_to)
        else:
            mae, _ = trainer.val(
                model=trainer.model, test_data_loader=loader,
                accelerator=trainer.accelerator, thresholding=True, save_to=save_to)

        trainer.accelerator.wait_for_everyone()
        logger.info(f"{dataset_name} | MAE: {mae:.6f} | time: {time.time()-t0:.1f}s")

        if trainer.accelerator.is_main_process:
            from utils.eval import eval
            eval_score = eval(mask_path=mask_path, pred_path=cfg.results_folder, dataset_name=dataset_name)
            if eval_score:
                for metric, value in eval_score.items():
                    logger.info(f"  {metric}: {value:.6f}")
                all_results[dataset_name] = eval_score

        trainer.accelerator.wait_for_everyone()

    logger.info(f"Total time: {(time.time()-total_start)/60:.1f} min")

    if trainer.accelerator.is_main_process and all_results:
        results_file = Path(cfg.results_folder) / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {results_file}")