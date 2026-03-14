from utils import init_env
import argparse
import torch
import logging
import os
from datetime import datetime
from pathlib import Path
from utils.collate_utils import collate, SampleDataset
from utils.import_utils import instantiate_from_config, recurse_instantiate_from_config, get_obj_from_str
from utils.init_utils import add_args, config_pretty
from utils.train_utils import set_random_seed
from torch.utils.data import DataLoader
from utils.trainer import Trainer

set_random_seed(42)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

def calculate_flops(model, input_size, device='cuda', batch_size=1):
    try:
        from thop import profile, clever_format
        
        model = model.to(device)
        model.eval()
        dummy_x = torch.randn(batch_size, 3, input_size, input_size).to(device)
        dummy_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        dummy_cond = torch.randn(batch_size, 3, input_size, input_size).to(device)

        try:
            flops, params = profile(model, inputs=(dummy_x, dummy_timesteps, dummy_cond), verbose=False)
        except:
            try:
                flops, params = profile(model, inputs=(dummy_x, dummy_timesteps), verbose=False)
            except:
                flops, params = profile(model, inputs=(dummy_x,), verbose=False)
        
        flops, params = clever_format([flops, params], "%.3f")
        return flops, params
    except ImportError:
        return "N/A (install thop: pip install thop)", "N/A"
    except Exception as e:
        return f"Error: {str(e)}", "N/A"

def setup_file_logger(results_folder=None, project_name="CamoDiffusion"):
    """设置logger"""
    if results_folder is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = f"./results/{project_name}_{timestamp}"
    
    results_path = Path(results_folder)
    results_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_file = results_path / 'training.log'
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("="*50)
    logger.info(f"Training started at {datetime.now()}")
    logger.info(f"Results will be saved to: {results_path}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*50)
    
    return logger, str(results_path)

def get_loader(cfg):
    train_dataset = instantiate_from_config(cfg.train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers)

    test_dataset = SampleDataset(full_dataset=instantiate_from_config(cfg.test_dataset.USOD10K), interval=10)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate
    )
    return train_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--results_folder', type=str, default=None, help='Folder to save results and logs.')
    parser.add_argument('--num_epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--calculate_flops', action='store_true', help='Calculate FLOPs (requires thop)')
    parser.add_argument('--input_size', type=int, default=352, help='Input image size for FLOPs calculation')
    
    cfg = add_args(parser)
    config_pretty(cfg)
    project_name = getattr(cfg, 'project_name', 'CamoDiffusion')
    logger, results_folder = setup_file_logger(cfg.results_folder, project_name)
    logger.info("Configuration:")
    logger.info(f"  Epochs: {cfg.num_epoch}")
    logger.info(f"  Batch size: {cfg.batch_size}")
    logger.info(f"  Gradient accumulation: {cfg.gradient_accumulate_every}")
    logger.info(f"  Learning rate min: {cfg.lr_min}")
    logger.info(f"  FP16: {cfg.fp16}")
    logger.info(f"  Workers: {cfg.num_workers}")
    cond_uvit = instantiate_from_config(cfg.cond_uvit,
                                        conditioning_klass=get_obj_from_str(cfg.cond_uvit.params.conditioning_klass))
    model = recurse_instantiate_from_config(cfg.model,
                                            unet=cond_uvit)
    diffusion_model = instantiate_from_config(cfg.diffusion_model,
                                              model=model)
    
    # ========== 统计模型参数量 ==========
    logger.info("\n" + "="*50)
    logger.info("Model Statistics:")
    logger.info("="*50)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total Parameters: {format_number(total_params)} ({total_params:,})")
    logger.info(f"Trainable Parameters: {format_number(trainable_params)} ({trainable_params:,})")
    logger.info(f"Non-trainable Parameters: {format_number(total_params - trainable_params)} ({total_params - trainable_params:,})")
    param_size = total_params * 4 / (1024 ** 2)
    logger.info(f"Model Size (FP32): {param_size:.2f} MB")
    # ========== 计算FLOPs==========
    if cfg.calculate_flops:
        logger.info("\nCalculating FLOPs (this may take a while)...")
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_size = cfg.input_size
            flops, params = calculate_flops(model, input_size, device, batch_size=1)
            logger.info(f"FLOPs: {flops}")
            logger.info(f"Params (from thop): {params}")
            if "Error" in str(flops):
                logger.info("\nManual FLOPs estimation:")
                logger.info("Note: Diffusion models typically require multiple forward passes during inference")
                single_pass_flops = total_params * 2 / 1e9
                logger.info(f"Single forward pass estimate: ~{single_pass_flops:.2f}G FLOPs")
                logger.info(f"For 50 diffusion steps: ~{single_pass_flops * 50:.2f}G FLOPs ({single_pass_flops * 50 / 1000:.2f}T FLOPs)")
                logger.info("(These are rough estimates. Actual FLOPs depend on model architecture)")
        except Exception as e:
            logger.warning(f"Failed to calculate FLOPs: {str(e)}")
            logger.warning("Install thop for FLOPs calculation: pip install thop")
    
    logger.info("="*50 + "\n")
    train_loader, test_loader = get_loader(cfg)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    optimizer = instantiate_from_config(cfg.optimizer, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epoch, eta_min=cfg.lr_min)
    

    trainer = Trainer(
        diffusion_model, train_loader, test_loader,
        train_val_forward_fn=get_obj_from_str(cfg.train_val_forward_fn),
        gradient_accumulate_every=cfg.gradient_accumulate_every,
        results_folder=results_folder,
        optimizer=optimizer, scheduler=scheduler,
        train_num_epoch=cfg.num_epoch,
        amp=cfg.fp16,
        log_with=None,
        cfg=cfg,
    )
    
    # 加载预训练模型或恢复训练
    if getattr(cfg, 'resume', None) or getattr(cfg, 'pretrained', None):
        logger.info(f"Loading model from resume: {cfg.resume}, pretrained: {cfg.pretrained}")
        trainer.load(resume_path=cfg.resume, pretrained_path=cfg.pretrained)

    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info(f"Training session ended at {datetime.now()}")
        logger.info("="*50)