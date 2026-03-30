### TRAINING TUTORIAL
## Table Of Content 
will be here 

This tutorial provides shorter and already evaluated information, which can also be found in the [README_TRAINING.md](/stm32ai-modelzoo-services/image_classification/docs/README_TRAINIG.md)
```
stm32ai-modelzoo-services/
  image_classification/
    docs/
      README_TRAINING.md
```
## 🗂Dataset Structure
Your dataset must follow this structure:
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
      lables_name.txt
```
## ⚙ Training Configuration (YAML)
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
   class_names: [bicycle, car] #all lables from lables_name.txt
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
Save file and move to the terminal.
```
cd path_to_stm32ai_modelzoo_services
source st_zoo311/bin/activate #activate virtual environment depending on system requirements
cd image_classification
python stm32ai_main.py --config-path ./config_file_examples/ --config-name training_config.yaml
```
After some time your model will be trained and saved into ./tf/src/experiments_outputs/ directory with all logs and visualized training results and matrixes

## ⚙ Quantization Configuration (YAML)
More detailed information is provided in [README_QUANTIZATION.md](/stm32ai-modelzoo-services/image_classification/docs/README_QUANTIZATION.md). In case of using guide to generate custom model, please refer to developers tutorial, which contains more detailed information about cutomization each parameter.
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
