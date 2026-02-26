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
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse

from clearml import Task
from clearml.backend_config.defs import get_active_config_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from api import get_model, get_dataloaders, get_predictor, get_trainer, get_evaluator, get_quantizer
from common.utils import mlflow_ini, set_gpu_memory_limit, get_random_seed, display_figures, log_to_file
from common.benchmarking import benchmark, cloud_connect
from common.prediction import gen_load_val_predict
from common.evaluation import gen_load_val
from semantic_segmentation.tf.src.utils import get_config
from semantic_segmentation.tf.src.deployment import deploy, deploy_mpu


# This function turns Tensorflow's eager mode on and off.
# Eager mode is for debugging the Model Zoo code and is slower.
# Do not set argument to True to avoid runtime penalties.
tf.config.run_functions_eagerly(False)


def _process_mode(configs: DictConfig = None) -> None:
    """
    Process the selected mode of operation.

    Args:

        configs: configuration object.

    Returns:
        None
    """
    mlflow.log_param("model_path", configs.model.model_path)
    mode = configs.operation_mode
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    model = get_model(cfg=configs)
    print(f"[INFO] : Model loaded from {model.model_path} with input shape {configs.model.input_shape}")

    if mode not in ['benchmarking', 'deployment']:
        dataloaders = get_dataloaders(configs)

    if mode == 'training':
        trainer = get_trainer(cfg=configs, model=model,
                              dataloaders=dataloaders)
        trained_model = trainer.train()
        display_figures(configs)
        evaluator = get_evaluator(cfg=configs, model=trained_model, dataloaders=dataloaders)
        evaluator.evaluate()
    elif mode == 'evaluation':
        gen_load_val(cfg=configs)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        evaluator = get_evaluator(cfg=configs, model=model,
                              dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
    elif mode == 'deployment':
        if configs.hardware_type == "MPU":
            print("MPU_DEPLOYMENT")
            deploy_mpu(cfg=configs)
        else:
            deploy(cfg=configs)
        print('[INFO] : Deployment complete.')
        if configs.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'quantization':
        quantizer = get_quantizer(cfg=configs, model=model, dataloaders=dataloaders)
        quantizer.quantize()
    elif mode == 'prediction':
        gen_load_val_predict(cfg=configs)
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        predictor = get_predictor(cfg=configs, model=model, dataloaders=dataloaders)
        predictor.predict()
    elif mode == 'benchmarking':
        benchmark(cfg=configs, model_path_to_benchmark=model.model_path)
        print('[INFO] : Benchmarking complete.')
        
    elif mode == 'chain_tqe':
        trainer = get_trainer(cfg=configs, model=model, dataloaders=dataloaders)
        trained_model = trainer.train()
        evaluator = get_evaluator(cfg=configs, model=trained_model, dataloaders=dataloaders)
        evaluator.evaluate()
        quantizer = get_quantizer(cfg=configs, model=trained_model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=configs, model=quantized_model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        print('[INFO] : chain_tqe complete.')

    elif mode == 'chain_tqeb':
        trainer = get_trainer(cfg=configs, model=model, dataloaders=dataloaders)
        trained_model = trainer.train()
        evaluator = get_evaluator(cfg=configs, model=trained_model, dataloaders=dataloaders)
        evaluator.evaluate()
        quantizer = get_quantizer(cfg=configs, model=trained_model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=configs, model=quantized_model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        credentials = None
        if configs.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=configs.tools.stedgeai.version)
        benchmark(cfg=configs, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_tqeb complete.')

    elif mode == 'chain_eqe':        
        evaluator = get_evaluator(cfg=configs, model=model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        quantizer = get_quantizer(cfg=configs, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=configs, model=quantized_model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        print('[INFO] : chain_eqe complete.')

    elif mode == 'chain_eqeb':
        evaluator = get_evaluator(cfg=configs, model=model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        quantizer = get_quantizer(cfg=configs, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        evaluator = get_evaluator(cfg=configs, model=quantized_model, dataloaders=dataloaders)
        evaluator.evaluate()
        display_figures(configs)
        credentials = None
        if configs.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=configs.tools.stedgeai.version)
        benchmark(cfg=configs, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_eqeb complete.')

    elif mode == 'chain_qb':
        quantizer = get_quantizer(cfg=configs, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        credentials = None
        if configs.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=configs.tools.stedgeai.version)
        benchmark(cfg=configs, model_path_to_benchmark=quantized_model.model_path, credentials=credentials)
        print('[INFO] : chain_qb complete.')

    elif mode == 'chain_qd':
        credentials = None
        if configs.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stedgeai_core_version=configs.tools.stedgeai.version)
        quantizer = get_quantizer(cfg=configs, model=model, dataloaders=dataloaders)
        quantized_model = quantizer.quantize()
        if configs.hardware_type == "MCU":
            deploy(cfg=configs, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        else:
            print("MPU DEPLOYMENT")
            deploy_mpu(cfg=configs, model_path_to_deploy=quantized_model.model_path, credentials=credentials)
        print('[INFO] : Deployment complete.')
        if configs.deployment.hardware_setup.board == "STM32N6570-DK":
            print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
        print('[INFO] : chain_qd complete.')

    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir)
    if mode in ['benchmarking', 'chain_qb', 'chain_eqeb', 'chain_tqeb']:
        if configs.tools.stedgeai.on_cloud:
            mlflow.log_param("stedgeai_core_version", configs.tools.stedgeai.version)
        else:
            mlflow.log_param("stedgeai_core_version", configs.tools.stedgeai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')

    # ClearML - Example how to get task's context anywhere in the file.
    # Checks if there's a valid ClearML configuration file
    if get_active_config_file() is not None: 
        print(f"[INFO] : ClearML task connection")
        task = Task.current_task()
        task.connect(configs)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] : Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='semseg_modelzoo_task')
        # ClearML - Optional yaml logging 
        task.connect_configuration(name=cfg.operation_mode, 
                                   configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    _process_mode(configs=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
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
