## Live FFT + Spectrogram Script.py
### Example Output 
![Py output](/docs/img/py_example.png)
### Purpose
This script provides **real-time visualization and analysis** of I/Q data received from an STM32 over UART.  
It displays:
- Time-domain signal  
- Frequency spectrum (FFT)  
- Spectrogram (STFT)  
- Doppler-based velocity estimation  

---

### How It Works
- **Serial thread:** Reads and parses incoming data  
- **Processing thread:** Computes FFT, spectrogram, and Doppler  
- **GUI:** Updates plots in real time using matplotlib  

---

### Key Parameters

#### Communication
- `PORT` – Serial port (e.g. `/dev/tty.usbmodem1102`)  
- `BAUDRATE` – Must match STM32  

#### Signal Processing
- `SAMPLING_RATE` – Defines frequency axis  
- `FFT_LEN` – FFT size (resolution vs speed)  
- `HOP` – Spectrogram time resolution  

#### Buffers
- `BUFFER_SIZE` – Time-domain window length  
- `WATERFALL_ROWS` – Spectrogram history  

#### Scaling
- `FSR`, `STEP` – Convert ADC values to voltage  

#### Visualization
- `UPDATE_MS` – Plot refresh rate  
- `cmin / cmax` – Spectrogram color range  

---

### Output
- Real-time signal plots  
- Dominant frequency detection  
- Doppler frequency (`fd`)  
- Estimated velocity (`v`) and direction  

---

### Summary
A real-time tool for **signal analysis and Doppler estimation** using STM32-acquired I/Q data.

---
---

## Live FFT on STM32 FFT_SCRIPT.PY
![Py output2](/docs/img/fft-on-stm1.png)

