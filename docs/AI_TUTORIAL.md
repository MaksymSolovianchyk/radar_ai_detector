
## Table Of Content 
<details open>
<summary><b>Table Of Content</b></summary>

- <a href="#deployment-tutorial"><b>Deployment Tutorial</b></a>
  - <a href="#software-requirements">Software Requirements</a>
  - <a href="#repository-setup">Repository Setup</a>
  - <a href="#install-python-dependencies-macos">Install Python Dependencies</a>
  - <a href="#optional-nvidia-gpu-setup-with-conda">NVIDIA GPU Setup</a>
  - <a href="#run-deployment">Run Deployment</a>

- <a href="#training-tutorial"><b>Training Tutorial</b></a>
  - <a href="#dataset-structure">Dataset Structure</a>
  - <a href="#training-configuration-yaml">Training Configuration</a>
  - <a href="#training-output">Training Output</a>
  - <a href="#what-is-the-next-step">Next Step</a>

- <a href="#quantization-configuration-yaml"><b>Quantization</b></a>

- <a href="#troubleshooting"><b>Troubleshooting</b></a>

</details>

---
This tutorial provides a simplified and validated overview of the workflow, which can also be found in the [README_TRAINING.md](/stm32ai-modelzoo-services/image_classification/docs/README_TRAINING.md). All console commands are applicable for other stm32ai projects, do not forget to change YAML file.
```
stm32ai-modelzoo-services/
  image_classification/
    docs/
      README_TRAINING.md
```
## Deployment Tutorial
<a id="deployment-tutorial"></a>
### To compile the image_classification project
1.	Software requirements:
<a id="software-requirements"></a>
The following software tools are required:
	•	STEdgeAI Core - Used to convert a trained .tflite or .onnx AI model into optimized C code for STM32 microcontrollers.
	•	STM32CubeIDE - Used to build the embedded C project and flash the firmware to the STM32 board.
	•	Python 3.x - Required to run the STM32 AI Model Zoo Services scripts.

2.	 Repository Setup
<a id="repository-setup"></a>
Clone the following two repositories:
	•	stm32ai-modelzoo
	•	stm32ai-modelzoo-services
After cloning, navigate to the services repository:
```
cd stm32ai-modelzoo-services
```
3.	Install Python Dependencies (MacOS)
<a id="install-python-dependencies-macos"></a>
 ```terminal
brew install python@3.11                  
/opt/homebrew/bin/python3.11 -m venv st_zoo311
source st_zoo311/bin/activate
python --version   # should show 3.11.x
python -m pip install -U pip
python -m pip install "tensorflow==2.18.0"
pip install -r requirements.txt
```
4.	(Optional) NVIDIA GPU Setup with Conda
<a id="optional-nvidia-gpu-setup-with-conda"></a>
If using an NVIDIA GPU and Conda, install CUDA dependencies:
```
conda install -c conda-forge cudatoolkit=11.8 cudnn

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
5.	Navigate to the Desired Project
`cd image_classification`

Open deployment_n6_config.yaml and modify it according to the example below. Verify model_path with desired model that you want to deploy (in model-zoo directory), path_to_stdegai and path_to_cubeIDE must be changed according to your file location. In case you have STM AI studio, you can find paths there in settings (Windows).

```yaml
model:
  # path to a `.tflite` or `.onnx` file.
  model_path: ../../stm32ai-modelzoo/image_classification/mobilenetv2/ST_pretrainedmodel_public_dataset/tf_flowers/mobilenetv2_a035_128_fft/mobilenetv2_a035_128_fft_int8.tflite
operation_mode: deployment
dataset:
  dataset_name: tf_flowers
  class_names: [daisy, dandelion, roses, sunflowers, tulips]
  classes_file_path: ./datasets/labels_tf_flowers.txt

preprocessing:
  resizing:
    interpolation: bilinear
    aspect_ratio: crop
  color_mode: rgb # rgb, bgr

tools:
  stedgeai:
    optimization: balanced
    on_cloud: False
    path_to_stedgeai: /Applications/ST/STEdgeAI/3.0/Utilities/mac/stedgeai
  path_to_cubeIDE: /Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE

deployment:
  c_project_path: ../application_code/image_classification/STM32N6/
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32N6
    board: STM32N6570-DK

hydra:
  run:
    dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

mlflow:
  uri: ./tf/src/experiments_outputs/mlruns
```

6.	Connect board to the PC and then the python script has to be compiled via terminal using command:
```
python3 stm32ai_main.py --config-path ./config_file_examples/ --config-name deployment_n6_config.yaml
```
In case of any errors related to 'Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.' (fix for MacOS)
```
export PATH="/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin:$PATH"
hash -r
export PATH="/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.macos64_1.0.100.202509120712/tools/bin:$PATH"
hash -r
```
7.	Next step is to open STMCubeIDE and open existing project in directory
stm-model-zoo-services -> application_code -> image_classification -> STM32N6
8.	In STM Cube IDE navigate to project and run it with Run button. After building and compiling, the image from the camera will be available on the display.
<a id="run-deployment"></a>

## TRAINING TUTORIAL
<a id="training-tutorial"></a>
### 🗂Dataset Structure
cYour dataset must follow this structure:
```
vehicle-10/
boat/
        img1.jpg
        img2.jpg
    bicycle/
    helicopter/
    truck/
    minibus/
    train/
    car/
    bus/
    motorcycle/
    taxi/
```

All lables that are mentined in the dataset should be represented in .txt file in directory:
```
stm32ai-modelzoo-services/
  image_classification/
    datasets/
      labels_name.txt
```
## ⚙ Training Configuration (YAML)
<a id="training-configuration-yaml"></a>
The training_config.yaml is located in:
```
stm32ai-modelzoo-services/
  image_classification/
    config_file_examples/
      training_config.yaml
```
Here is example configuration to train it with vehicle-10 dataset installed from the internet

```yaml
general:
  project_name: training_test
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 127
  gpu_memory_limit: 3 #adjust according to available memory on PC

operation_mode: training

model:
   model_name: mobilenetv2_a035 #select here desired model from atm32ai-modelzoo availabe models list
   input_shape: (224, 224, 3)
   pretrained: True

dataset:
   dataset_name: custom_dataset
   class_names: [bicycle, car] #all lables from labels_name.txt
   training_path: /Users/maksym.solovianchyk/Downloads/vehicle-10
   validation_path:
   validation_split: 0.15
   test_path:

preprocessing:
   rescaling:
      scale: 1/127.5
      offset: -1
   resizing:
      aspect_ratio: fit
      interpolation: nearest
   color_mode: rgb

data_augmentation:
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.2
  random_flip:
    mode: horizontal_and_vertical
  random_translation:
    width_factor: 0.2
    height_factor: 0.2
  random_rotation:
    factor: 0.15
  random_zoom:
    width_factor: 0.25
    height_factor: 0.25

training:
   batch_size: 64
   epochs: 200 #can be adjusted depending on desired accuracy
   dropout: 0.3
   optimizer:
      Adam:
         learning_rate: 0.001
   callbacks:
      ReduceLROnPlateau:
         monitor: val_accuracy
         factor: 0.5
         patience: 10
      EarlyStopping:
         monitor: val_accuracy
         patience: 30 

mlflow:
   uri: ./tf/src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S} 
```
### 📁 Training Output
<a id="training-output"></a>
```
tf/src/experiments_outputs/YYYY_MM_DD_HH_MM_SS/
```
Inside:
```
saved_models/
    best_model.keras
    best_augmented_model.keras
    last_augmented_model.keras
```
### 🎯 Which Model to Use?

```
best_model.keras
```
### 🤖 What Is The Next Step?
<a id="what-is-the-next-step"></a>
Save file and move to the terminal.
```
cd path_to_stm32ai_modelzoo_services
source st_zoo311/bin/activate #activate virtual environment depending on system requirements
cd image_classification
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
After some time your model will be trained and saved into ./tf/src/experiments_outputs/ directory with all logs and visualized training results and matrixes

## ⚙ Quantization Configuration (YAML)
<a id="quantization-configuration-yaml"></a>
More detailed information is provided in [README_QUANTIZATION.md](/stm32ai-modelzoo-services/image_classification/docs/README_QUANTIZATION.md). In case of using guide to generate custom model, please refer to developers tutorial, which contains more detailed information about cutomization each parameter. As a result, you will get a working trained model.
Example content of YAML configuration file 
```yaml
model:
   model_path: /Users/maksym.solovianchyk/Documents/Graduation_Internship/radar_ai_detector/stm32ai-modelzoo-services/image_classification/tf/src/experiments_outputs/2026_03_30_15_47_20/saved_models/best_model.keras 
operation_mode: quantization

dataset:
  dataset_name: custom_dataset #do not change if using custom dataset
  class_names: [hand_wave, idle] #adjust according to your classes
  quantization_path: /Users/maksym.solovianchyk/Documents/Graduation_Internship/radar_ai_detector/Recorded_data/tf_dataset

preprocessing:
   rescaling:
      scale: 1/255 #change according to model specifications
      offset: 0 #change according to model specifications
   resizing:
      aspect_ratio: fit
      interpolation: nearest
   color_mode: grayscale #change according to model specifications

quantization:
   quantizer: TFlite_converter
   quantization_type: PTQ
   quantization_input_type: uint8
   quantization_output_type: int8 #change according to model specifications
   export_dir: quantized_models

mlflow:
   uri: ./tf/src/experiments_outputs/mlruns

hydra:
   run:
      dir: ./tf/src/experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}

```
## ⚙ Troubleshooting
<a id="troubleshooting"></a>
During quantization or training, many problems may occur due to incorrect settings, that are not supported by pre-trained model. To check requirements go to [image_classification repository](/stm32ai-modelzoo/image_classification). 
Solution: generate a custom model by using a [custom model py](/stm32ai-modelzoo-services/image_classification/tf/src/models/custom_model.py). Here you apply all customized settings and then use it in the training configuration file.
