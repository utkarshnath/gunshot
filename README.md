# Setup Instructions

Please select Raspberry pi OS lite (64 bit) as the Operating System
```
sudo apt-get update
sudo apt-get upgrade
sudo reboot
sudo apt-get install python3-pip
sudo pip3 install --upgrade adafruit-python-shell --break-system-packages
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/i2smic.py
sudo python3 i2smic.py
sudo reboot
```
```
add the following two statements to this file /boot/firmware/config.txt 
--> dtoverlay=googlevoicehat-soundcard
--> enable_uart=1
sudo reboot
```
```
sudo apt-get install libatlas-base-dev libportaudio2 libasound-dev
python3 -m venv my_env
source my_env/bin/activate

python3 -m pip install sounddevice
python3 -m pip install scipy
pip install librosa
sudo apt install libsndfile1 libsndfile1-dev
pip install soundfile --force-reinstall
pip install pyserial
```

# Commands to run the program
```
source my_env/bin/activate
cd gunshot_project
python run.py
```

# Serial output
The Raspberry Pi has a UART interface available on its GPIO pins, specifically GPIO 14 (TXD) and GPIO 15 (RXD).
```
Connect the Raspberry Pi GPIO pins to your serial device:
GPIO 14 (TXD) -> Device RX
GPIO 15 (RXD) -> Device TX
GND -> GND
```


Sample python script to send serial data:
```
import serial
import time

# Configure the serial port
ser = serial.Serial('/dev/serial0', 9600, timeout=1)  # Use /dev/serial0 for GPIO UART
ser.flush()

while True:
    ser.write(b"Hello, Serial Output!\n")  # Send data
    print("Message sent!")
    time.sleep(1)  # Wait 1 second between messages
```

Please uncomment line number 16, 17, 102, 103, 131 and 132 in run.py to send serial output. 
Please note, these commands won't run without attaching a serial device. 

The serial output will be 1 if a gunshot is detected.









