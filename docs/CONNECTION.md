## Connection for Radar sensor with STM32N6570-DK via STMod+
[STM32N6570-DK Datasheet](https://www.st.com/resource/en/user_manual/um3300-discovery-kit-with-stm32n657x0-mcu-stmicroelectronics.pdf)

| Radar PCB Pin | STM32 Pin | Signal | Description |
|---------------|-----------|--------|-------------|
| VIN | 6 | +5V | Power supply |
| GND | 5 | GND | Ground reference |
| SPI2_CLK | 4 | SPI5_SCK | SPI clock |
| SPI2_MISO | 3 | SPI5_MISO | Data from ADC to STM32 |
| SPI2_MOSI | 2 | SPI5_MOSI | Data from STM32 to ADC |
| ADC_SPI_CS1 | 1 | SPI5_CS | Chip select |
| ADC_DRDY | 11 | INT (PC11) | Data ready interrupt from ADC → MCU (best practice for sampling) |
| ADC_RESET | 12 | RESET (PB3) | Lets MCU reset/sync the ADC. (Using PB3 is convenient; a normal GPIO also works.) |

![STMod+ Connector Pinout](img/stmod_connector_pinout.jpg)
![Radar Sensor Pinout](img/radar_pinout.jpg)
