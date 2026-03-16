#targeting python 3.13

import time
import serial
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.widgets import Button, RadioButtons
import matplotlib.patches as patches
from numpy import trapezoid

ser = serial.Serial("/dev/tty.usbmodem1102", baudrate=576000, timeout=1)

timestamps = []
global_max = 10
sampling_rate = 3904
doppler_freq = 0.0
velocity = 0.0
direction = "none"
fft_energy = 0.0


decim = 5
FSR = 0.15      #0.009375#0.15
STEP = FSR / (2**23)

# buffer setup
BUFFER_SIZE = 5000
ch1_data = np.zeros(BUFFER_SIZE)
ch2_data = np.zeros(BUFFER_SIZE)
write_idx = 0
lock = threading.Lock()

# FFT n waterfall
FFT_LEN = 1024
fft_len_options = [256, 512, 1024, 2048, 4096]
fft_len_lock = threading.Lock()
WATERFALL_HISTORY = 200
waterfall = np.full((WATERFALL_HISTORY, FFT_LEN), -120.0)  
fft_slice = np.full(FFT_LEN, -120.0)  
fft_lock = threading.Lock()
EPSILON = 1e-12 #avoid log(0)
cor_fac = 1.59 #for hamming window

clim_min = -100.0
clim_max = 0.0

def bytes_to_int24(b):
    val = b[0] << 16 | b[1] << 8 | b[2]
    if val & 0x800000:
        val -= 1 << 24
    return val

def reader_thread():
    global write_idx, ch1_data, ch2_data
    leftover = b""
    while True:         
        try:
            buf = ser.read(ser.in_waiting or 64)
        except serial.SerialException as e:
            print("serial error:", e)
            time.sleep(1)
            continue

        if not buf:
            time.sleep(0.01)
            continue
        buf = leftover + buf
        i = 0
        while i + 8 <= len(buf):
            frame = buf[i:i+8]
            if frame[6:8] == b"\r\n":
                ch1 = bytes_to_int24(frame[0:3]) * STEP
                ch2 = bytes_to_int24(frame[3:6]) * STEP
                with lock:
                    ch1_data[write_idx] = ch1
                    ch2_data[write_idx] = ch2
                    write_idx = (write_idx + 1) % BUFFER_SIZE
                i += 8
            else:
                i += 1      #if no lock skip a byte to resync
        leftover = buf[i:]

def fft_thread():
    global fft_slice, global_max, doppler_freq, velocity, direction, fft_energy
     
    c = 3e8
    fc = 24e9  #radar freq
    win_cache={}
    while True:
        with fft_len_lock:
            fft_len = FFT_LEN
        if fft_len not in win_cache:
            win_cache[fft_len] = np.hamming(fft_len)
        win = win_cache[fft_len]

        with lock:
            # wait for enough samples for fft
            if write_idx < fft_len and write_idx != 0:
                data_ready = False #if not enough skip
                pass
            else:
                data_ready = True
                # build index for last FFT_LEN samples
                idx = np.arange(write_idx - FFT_LEN, write_idx) % BUFFER_SIZE
                i_data = ch2_data[idx].copy()
                q_data = ch1_data[idx].copy()
                

        if not data_ready:
            time.sleep(0.05)
            continue
        
        sig = i_data + 1j * q_data          # use both i imaginary and q real to make a phasor

        windowed = sig * win

        fft_data = np.fft.fft(windowed, n=fft_len)  
        fft_data_shifted = np.fft.fftshift(fft_data)    #use fftshift to center 0Hz
        fft_mag = np.abs(fft_data_shifted) + EPSILON
        fft_mag_pure = np.abs(fft_data_shifted)
        df = sampling_rate / fft_len
        fft_energy = (np.trapezoid(fft_mag_pure**2,dx=df)/ fft_len**2)*cor_fac # binn width: df = fs / N 

        local_max = np.max(fft_mag) #for the colors and dB, todo:calibrate to absolute value
        if local_max > global_max:
            global_max = local_max
            

        fft_db = 20.0 * np.log10(fft_mag / (global_max + EPSILON))
        #freqs = np.fft.fftfreq(FFT_LEN, d=1.0 / sampling_rate)

        phase = np.unwrap(np.angle(sig))    #phasor
        dphase = np.diff(phase)
        # Hz = derivative(phase)/(2*pi) * fs
        inst_freq = (dphase / (2.0 * np.pi)) * sampling_rate
        if inst_freq.size > 0:
            doppler_freq = np.median(inst_freq)
        else:
            doppler_freq = 0.0

        velocity = (doppler_freq * c) / (2.0 * fc)

        direction = "approaching" if doppler_freq > 0 else ("receding" if doppler_freq < 0 else "none")

        fft_db_clipped = np.clip(fft_db, -120.0, 10.0)

        with fft_lock:
            np.copyto(fft_slice, fft_db_clipped)

        time.sleep(0.01)    

SERVER_ADDR = ("127.0.0.1", 5005)

CRC8Table = [
    0x00, 0x97, 0xb9, 0x2e, 0xe5, 0x72, 0x5c, 0xcb,
    0x5d, 0xca, 0xe4, 0x73, 0xb8, 0x2f, 0x01, 0x96,
    0xba, 0x2d, 0x03, 0x94, 0x5f, 0xc8, 0xe6, 0x71,
    0xe7, 0x70, 0x5e, 0xc9, 0x02, 0x95, 0xbb, 0x2c,
    0xe3, 0x74, 0x5a, 0xcd, 0x06, 0x91, 0xbf, 0x28,
    0xbe, 0x29, 0x07, 0x90, 0x5b, 0xcc, 0xe2, 0x75,
    0x59, 0xce, 0xe0, 0x77, 0xbc, 0x2b, 0x05, 0x92,
    0x04, 0x93, 0xbd, 0x2a, 0xe1, 0x76, 0x58, 0xcf,
    0x51, 0xc6, 0xe8, 0x7f, 0xb4, 0x23, 0x0d, 0x9a,
    0x0c, 0x9b, 0xb5, 0x22, 0xe9, 0x7e, 0x50, 0xc7,
    0xeb, 0x7c, 0x52, 0xc5, 0x0e, 0x99, 0xb7, 0x20,
    0xb6, 0x21, 0x0f, 0x98, 0x53, 0xc4, 0xea, 0x7d,
    0xb2, 0x25, 0x0b, 0x9c, 0x57, 0xc0, 0xee, 0x79,
    0xef, 0x78, 0x56, 0xc1, 0x0a, 0x9d, 0xb3, 0x24,
    0x08, 0x9f, 0xb1, 0x26, 0xed, 0x7a, 0x54, 0xc3,
    0x55, 0xc2, 0xec, 0x7b, 0xb0, 0x27, 0x09, 0x9e,
    0xa2, 0x35, 0x1b, 0x8c, 0x47, 0xd0, 0xfe, 0x69,
    0xff, 0x68, 0x46, 0xd1, 0x1a, 0x8d, 0xa3, 0x34,
    0x18, 0x8f, 0xa1, 0x36, 0xfd, 0x6a, 0x44, 0xd3,
    0x45, 0xd2, 0xfc, 0x6b, 0xa0, 0x37, 0x19, 0x8e,
    0x41, 0xd6, 0xf8, 0x6f, 0xa4, 0x33, 0x1d, 0x8a,
    0x1c, 0x8b, 0xa5, 0x32, 0xf9, 0x6e, 0x40, 0xd7,
    0xfb, 0x6c, 0x42, 0xd5, 0x1e, 0x89, 0xa7, 0x30,
    0xa6, 0x31, 0x1f, 0x88, 0x43, 0xd4, 0xfa, 0x6d,
    0xf3, 0x64, 0x4a, 0xdd, 0x16, 0x81, 0xaf, 0x38,
    0xae, 0x39, 0x17, 0x80, 0x4b, 0xdc, 0xf2, 0x65,
    0x49, 0xde, 0xf0, 0x67, 0xac, 0x3b, 0x15, 0x82,
    0x14, 0x83, 0xad, 0x3a, 0xf1, 0x66, 0x48, 0xdf,
    0x10, 0x87, 0xa9, 0x3e, 0xf5, 0x62, 0x4c, 0xdb,
    0x4d, 0xda, 0xf4, 0x63, 0xa8, 0x3f, 0x11, 0x86,
    0xaa, 0x3d, 0x13, 0x84, 0x4f, 0xd8, 0xf6, 0x61,
    0xf7, 0x60, 0x4e, 0xd9, 0x12, 0x85, 0xab, 0x3c
]

def crc8(data, init=0x00):
    crc = init
    for byte in data:
        crc = CRC8Table[crc ^ byte]
    return crc

def forward_to_serial(data_bytes):
    crc = crc8(data_bytes, init=0x00)
    packet = data_bytes + bytes([crc])
    try:
        ser.write(packet)
    except Exception as e:
        print("Serial error:", e)

def socket_listener(forward_to_serial):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(SERVER_ADDR)
    server_sock.listen(1)
    print(f"listening for GUI commands on {SERVER_ADDR}...")

    while True:
        conn, addr = None, None
        try:
            print("waiting for connection...")
            conn, addr = server_sock.accept()
            print(f"GUI connected from {addr}")
            with conn:
                while True:
                    data = conn.recv(2)  # in bytes
                    if not data:
                        print("GUI disconnected")
                        break
                    if len(data) == 2:
                        cmd_id, value = data[0], data[1]
                        print(f"received command id={cmd_id:#02x}, value={value:#02x}")
                        forward_to_serial(bytes([cmd_id, value]))
                    else:
                        print(f"invalid command length: {len(data)}")
        except Exception as e:
            print("socket error:", e)
        finally:
            if conn:
                conn.close()
            time.sleep(1)

threading.Thread(target=socket_listener, args=(forward_to_serial,) ,daemon=True).start()
threading.Thread(target=reader_thread, daemon=True).start()
threading.Thread(target=fft_thread, daemon=True).start()

fig, (ax_time, ax_fftline, ax_fft) = plt.subplots(3, 1, figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)

x_time = np.arange(0, BUFFER_SIZE, decim)
line1, = ax_time.plot(x_time, np.zeros_like(x_time), label="Channel 1 (I)")
line2, = ax_time.plot(x_time, np.zeros_like(x_time), label="Channel 2 (Q)")
ax_time.legend()
ax_time.set_ylim(-FSR, FSR)
ax_time.set_title("time domain")

ax_min = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_scale = plt.axes([0.15, 0.05, 0.65, 0.03])

# waterfall
im = ax_fft.imshow(
    waterfall,
    aspect='auto',
    origin='lower',
    extent=[-sampling_rate/2, sampling_rate/2, 0, WATERFALL_HISTORY],
    cmap='inferno',
    vmin=clim_min,
    vmax=clim_max,
    interpolation='nearest'
)
fig.colorbar(im, ax=ax_fft, label="amplitude (dB)")
slider_min = Slider(ax_min, 'min (dB)', -120, 0, valinit=-100, valstep=1)
slider_scale = Slider(ax_scale, 'max (dB)', -120, 0, valinit=-10, valstep=1)

# fft line centered
line_fft, = ax_fftline.step(
    np.linspace(-sampling_rate/2, sampling_rate/2, FFT_LEN), 
    np.full(FFT_LEN, -120.0),                                
    where="mid",
    color="gold"
)
ax_fftline.set_title("fft slice")
ax_fftline.set_xlim(-sampling_rate/2, sampling_rate/2)
ax_fftline.set_ylim(-120, 10)
ax_fftline.set_ylabel("amplitude (dB)")

# doppler info text
info_text = ax_time.text(0.02, 0.9, "", transform=ax_time.transAxes, fontsize=12,
                         bbox=dict(facecolor='white', alpha=0.7))

ax_fftselect = plt.axes([0.90, 0.25, 0.10, 0.20]) 
radio_fft = RadioButtons(ax_fftselect, [str(l) for l in fft_len_options], active=2)  # default=1024

def set_fft_len(label):
    global FFT_LEN
    with fft_len_lock:
        FFT_LEN = int(label)
    print(f"FFT length changed to {FFT_LEN}")

radio_fft.on_clicked(set_fft_len)

def send_dc_block_512(event):
    forward_to_serial(bytes([3, 0x08]))

def send_dc_block_off(event):
    forward_to_serial(bytes([3, 0x00]))

axbutton_512 = plt.axes([0.9, 0.00, 0.10, 0.05])
button_512 = Button(axbutton_512, "DCBLOCK 512")
button_512.on_clicked(send_dc_block_512)

axbutton_off = plt.axes([0.9, 0.05, 0.10, 0.05])
button_off = Button(axbutton_off, "DCBLOCK OFF")
button_off.on_clicked(send_dc_block_off)

def update(frame):
    global waterfall, doppler_freq, velocity, direction, FFT_LEN, fft_slice
    with lock:
        idx = np.arange(write_idx, write_idx + BUFFER_SIZE) % BUFFER_SIZE
        y1 = ch1_data[idx][::decim]
        y2 = ch2_data[idx][::decim]
        line1.set_ydata(y1)
        line2.set_ydata(y2)

    with fft_len_lock:
        fft_len = FFT_LEN

    if waterfall.shape[1] != fft_len:
        waterfall = np.full((WATERFALL_HISTORY, fft_len), -120.0)
        with fft_lock:
            fft_slice =np.full(fft_len, -120.0)
        im.set_extent([-sampling_rate/2, sampling_rate/2, 0, WATERFALL_HISTORY])
        line_fft.set_xdata(np.linspace(-sampling_rate/2, sampling_rate/2, fft_len))
        line_fft.set_ydata(np.full(fft_len, -120.0))

    
    with fft_lock:
        fft_slice_copy = fft_slice.copy()

    line_fft.set_ydata(fft_slice_copy)

    waterfall[:-1] = waterfall[1:]  #shift rows
    waterfall[-1, :] = fft_slice_copy  #set slice
    im.set_data(waterfall)
    im.set_clim(slider_min.val, slider_scale.val)

    # doppler info text update
    info_text.set_text(f"{direction}\nfd = {doppler_freq:.1f} Hz\nv = {velocity:.2f} m/s, Energy={fft_energy:.2e}")

    return line1, line2, im, line_fft, info_text

ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
