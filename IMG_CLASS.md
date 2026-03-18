## To compile the image_classification proejct 
1.	Software requirements:
The following software tools are required:
	•	STEdgeAI Core - Used to convert a trained .tflite or .onnx AI model into optimized C code for STM32 microcontrollers.
	•	STM32CubeIDE - Used to build the embedded C project and flash the firmware to the STM32 board.
	•	Python 3.x - Required to run the STM32 AI Model Zoo Services scripts.

2.	 Repository Setup
Clone the following two repositories:
	•	stm32ai-modelzoo
	•	stm32ai-modelzoo-services
After cloning, navigate to the services repository:
```
cd stm32ai-modelzoo-services
```

4.	Python Environment Setup
Create a Virtual Environment
Create a Python virtual environment using venv:
```
python3 -m venv st_zoo
st_zoo\Scripts\activate.bat # – Windows
source st_zoo/bin/activate # - MacOS/Linux
```
4.	(Optional) NVIDIA GPU Setup with Conda
If using an NVIDIA GPU and Conda, install CUDA dependencies:
```
conda install -c conda-forge cudatoolkit=11.8 cudnn

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
5.	Install Python Dependencies
 ```
brew install python@3.11                  
/opt/homebrew/bin/python3.11 -m venv st_zoo311
source st_zoo311/bin/activate
python --version   # should show 3.11.x
python -m pip install -U pip
python -m pip install "tensorflow==2.18.0"
pip install -r requirements.txt
```
6.	Navigate to the Desired Project
`cd image_classification`

Open deployment_n6_config.yaml using a text editor or IDE and change file according to the example provided below. Verify model_path with desired model that you want to deploy (in model-zoo directory), path_to_stdegai and path_to_cubeIDE must be changed according to your file location. In case you have STM AI studio, you can find paths there in settings (Windows).

```
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

7.	Connect board to the PC and then the python script has to be compiled via terminal using command:
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
8.	Next step is to open STMCubeIDE and open existing project in directory
stm-model-zoo-services -> application_code -> image_classification -> STM32N6

 

9.	In STM Cube IDE navigate to project and run it with Run button. After building and compiling, the image from the camera will be available on the display.

