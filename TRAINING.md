### TRAINING TUTORIAL
This tutorial provides shorter and already evaluated information, that can also be found in the
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

```
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
