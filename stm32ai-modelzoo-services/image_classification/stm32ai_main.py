# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
from hydra.core.hydra_config import HydraConfig
import hydra
import warnings
from timm import utils as timm_utils
from types import SimpleNamespace

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import torch
from omegaconf import DictConfig
import mlflow
import argparse
import logging
from typing import Optional
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model, get_dataloaders, get_trainer, get_quantizer, get_evaluator, get_predictor
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.evaluation import gen_load_val
from common.prediction import gen_load_val_predict
from image_classification.tf.src.utils import get_config
from image_classification.tf.src.deployment import deploy, deploy_mpu
from common.onnx_utils.onnx_model_convertor import torch_model_export_static


def _process_mode(cfg: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:
        cfg (DictConfig): The configuration object.

    Returns:
        None
    Raises:
        ValueError: If an invalid operation_mode is selected or if required datasets are missing.
    """

    # Logging the operation_mode in the output_dir/stm32ai_main.log file
    mode = cfg.operation_mode
    mlflow.log_param("model_path", cfg.model.model_path)
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    # Connect to STM32Cube.AI Developer Cloud if needed
    credentials = None
    if cfg.tools and cfg.tools.stedgeai and cfg.tools.stedgeai.on_cloud:
        _, _, credentials = cloud_connect(stedgeai_core_version=cfg.tools.stedgeai.version)

    # Creates model
    model = get_model(cfg=cfg)
    saved_model_dir = os.path.join(cfg.output_dir, cfg.general.saved_models_dir)
    os.makedirs(saved_model_dir, exist_ok=True)
    if cfg.model.framework == 'torch' and isinstance(model, torch.nn.Module) and cfg.operation_mode not in ['training', 'chain_tb', 'chain_tqe', 'chain_tqeb', 'chain_tbqeb']:
        # Export Torch models in onnx format for all services but training 
        # (export to onnx is also handled at the end of the trainer.train() method)
        model = torch_model_export_static(cfg=cfg, 
                                          model_dir=saved_model_dir, 
                                          model=model)

    # Creates dataloaders
    dataloaders = get_dataloaders(cfg=cfg)

    # Executes Services
    if mode == 'training':
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(cfg)
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
    elif mode == 'evaluation':
        gen_load_val(cfg=cfg, model=model)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
    elif mode == 'deployment':
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg=cfg, model_path_to_deploy=model.model_path)
        else:
            deploy(cfg=cfg, model_path_to_deploy=model.model_path)
    elif mode == 'quantization':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
    elif mode == 'prediction':
        gen_load_val_predict(cfg=cfg, model=model)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predictor = get_predictor(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        predictor.predict()
    elif mode == 'benchmarking':
        benchmark(cfg=cfg, model_path_to_benchmark=model.model_path)
    elif mode == 'chain_tqe':
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        print('[INFO] : chain_tqe complete.')
    elif mode == 'chain_tqeb':
        trainer = get_trainer(cfg=cfg, 
                              model=model, 
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=trained_model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_tqeb complete.')
    elif mode == 'chain_eqe':
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        print('[INFO] : chain_eqe complete.')
    elif mode == 'chain_eqeb':
        evaluator = get_evaluator(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=cfg, 
                                  model=quantized_model, 
                                  dataloaders=dataloaders)
        acc = evaluator.evaluate()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_eqeb complete.')
    elif mode == 'chain_qb':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        benchmark(cfg=cfg, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qb complete.')
    elif mode == 'chain_qd':
        quantizer = get_quantizer(cfg=cfg, 
                                  model=model, 
                                  dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        if cfg.hardware_type == "MCU":
            deploy(cfg=cfg, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        else:
            deploy_mpu(cfg=cfg, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qd complete.')
    # Raise an error if an invalid mode is selected
    else: 
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(cfg.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        mlflow.log_param("stedgeai_core_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)
    
    # Logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(cfg)

def _fw_agnostic_initializations(cfg: DictConfig = None) -> DictConfig:
    """
    Framework-agnostic initializations.

    This function performs initializations that are independent of the specific deep learning framework being used.
    It includes parsing the configuration file, setting up MLFlow, and initializing ClearML if a valid configuration
    file is found.

    Args:
        cfg (DictConfig): Configuration object.

    Returns:
        DictConfig: Updated configuration object with initialized settings.
    """
    # Parse the configuration file and set the output directory
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir

    # Initialize MLFlow for experiment tracking
    # MLFlow is used to log metrics, parameters, and artifacts during training
    mlflow_ini(cfg)

    # Check if there's a valid ClearML configuration file and initialize ClearML
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        # If a ClearML configuration file is found, initialize ClearML
        print(f"[INFO] : ClearML initialization and configuration")
        # Initialize ClearML's Task object with the project and task names
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='ic_modelzoo_task')
        # Optionally log the configuration to ClearML
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)
    
    # Return the updated configuration object
    return cfg


def _tf_specific_initializations(cfg: DictConfig = None) -> None:
    """
    TensorFlow-specific initializations.

    This function performs initializations specific to TensorFlow, such as configuring GPU memory limits
    and setting a random seed for reproducibility.

    Args:
        cfg (DictConfig): Configuration object.
    """
    # Check if the 'general' section exists in the configuration
    if "general" in cfg and cfg.general:
        # Set an upper limit on GPU memory usage if specified in the configuration
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            # Warn the user if GPU memory usage is unlimited
            print("[WARNING] The usable GPU memory is unlimited.\n"
                "Please consider setting the 'gpu_memory_limit' attribute "
                "in the 'general' section of your configuration file.")

    # Set a random seed for reproducibility
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)


def _torch_specific_initializations(cfg: DictConfig = None) -> None:
    """
    PyTorch-specific initializations.

    This function is a placeholder for PyTorch-specific initializations, such as configuring GPU memory limits
    and setting a random seed for reproducibility.

    Args:
        cfg (DictConfig): Configuration object.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_args = SimpleNamespace(
        device=device,
    )
    device = timm_utils.init_distributed_device(temp_args)
    
    cfg.device = temp_args.device
    cfg.world_size = temp_args.world_size
    cfg.rank = temp_args.rank
    cfg.local_rank = temp_args.local_rank
    cfg.distributed = temp_args.distributed
    

@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    # Framework agnostic initializations
    cfg = _fw_agnostic_initializations(cfg)

    # Framework specific initializations
    if cfg.model.framework == "tf":
        _tf_specific_initializations(cfg)
    elif cfg.model.framework == "torch":
        _torch_specific_initializations(cfg)
    else:
        raise ValueError(f"Invalid framework used: {cfg.model.framework}")

    # Executes the required service
    _process_mode(cfg=cfg)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='./', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config.yaml', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
