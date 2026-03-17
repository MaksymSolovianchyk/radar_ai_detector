## PROJECT CONFIGURATION TUTORIAL IN CUBEMX
### SPI5 Configuration (Sensor data)
SPI5 is configured in the AppliSecure context.

- Mode: Master
- Virtual type: VM_MASTER
- Direction: 2 lines
- Data size: 8-bit
- Clock phase: 2nd edge
- NSS: Software
- Baud rate prescaler: 64
- Calculated baud rate: 1.5625 Mbit/s

SPI5 pins

- SCK: PE15
- MOSI: PG2
- MISO: PH8

  ![SPI5 Configuration](/docs/img/spi5_config.png)
### USART1 Configuration (Transmission data into terminal)

USART1 is configured in the AppliSecure context.
-	Mode: Asynchronous
- Baud rate: 576000
-	Word length: 8 bits
-	Parity: None
-	Oversampling: 8

USART1 pins
-	TX: PE5
-	RX: PE6

### DMA Configuration
Select GPDMA1 
Section <strong>GPDMA1 Mode and Configuration</strong> select Channel 0,1,2,3 <strong>Standard Request</strong>. In <strong>Configuration</strong> section 4 channels will appear <strong>CH0-3</strong>.
Configured simple request DMA channels:
-	Channel 0: SPI5_RX
-	Channel 1: SPI5_TX
-	Channel 2: USART1_RX
-	Channel 3: USART1_TX

DMA transfer properties:
Channel 0
-	Request: SPI5_RX
-	Direction: Peripheral to Memory
-	Source data width: Byte
-	Destination data width: Byte
-	Source Address increment: Disabled (Fixed will be in .ioc)
-	Source Allocated Port for Transfer: Port 1
-	Destination  Address increment: Enabled (Incremented will be in .ioc)
-	Dest Allocated Port for Transfer: Port 0
-	Circular mode: Disabled

Channel 1
-	Request: SPI5_TX
-	Direction: Memory to Peripheral
-	Source data width: Byte
-	Destination data width: Byte
- Source Address increment: Enabled (Incremented will be in .ioc)
-	Source Allocated Port for Transfer: Port 0
-	Destination  Address increment: Disabled (Fixed will be in .ioc)
-	Dest Allocated Port for Transfer: Port 1

Channel 2
-	Request: USART1_RX
-	Direction: Peripheral to Memory
-	Source data width: Byte
-	Destination data width: Byte
-	Source Address increment: Disabled (Fixed will be in .ioc)
-	Source Allocated Port for Transfer: Port 1
-	Destination  Address increment: Enabled (Incremented will be in .ioc)
-	Dest Allocated Port for Transfer: Port 0

Channel 3
-	Request: USART1_TX
-	Direction: Memory to Peripheral
- Source Address increment: Enabled (Incremented will be in .ioc)
-	Source Allocated Port for Transfer: Port 0
-	Destination  Address increment: Disabled (Fixed will be in .ioc)
-	Dest Allocated Port for Transfer: Port 1

### GPIO Configuration
  ![GPIO Configuration](/docs/img/gpio_config.png)
  
AppliSecure GPIO outputs

PA3
- Label: ADC_SPI_CS1
- Signal: GPIO_Output
- Default state: GPIO_PIN_SET
- Meaning: starts High

PB3
- Label: SYNC_RESET
- Signal: GPIO_Output
- Default state: GPIO_PIN_SET
PC9
- Label: PSUEN
- Signal: GPIO_Output

AppliSecure GPIO interrupt input
- PC11
- Signal: GPXTI11
- Mode: GPIO_MODE_IT_FALLING

### NVIC1_S_Application Configuration (NVIC)

AppliSecure interrupts enabled
- EXTI11_IRQn
- GPDMA1_Channel0_IRQn
- GPDMA1_Channel1_IRQn
- GPDMA1_Channel2_IRQn
- GPDMA1_Channel3_IRQn
- SPI5_IRQn
- USART1_IRQn
  
## COMMANDS
To see sensor output in terminal
`picocom --baud 576000 --parity n --databits 8 --stopbits 1 /dev/tty.usbmodem1102`
Where /dev/tty.usbmodem1102 is USB/USB-C port on PC
On macOS `ls /dev/tty.*`
On Windows `[System.IO.Ports.SerialPort]::getportnames()`
