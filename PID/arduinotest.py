import serial
import time
import threading

# arduino = serial.Serial(port='COM3', baudrate=9600, timeout=.1) # Windows 下
arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600, timeout=.1) # Linux/macOS 下

def send_value():
    while True:
        value = input("輸入要發送的數值: ")  # 從用戶獲取輸入
        message = f"C:{value}\n"  # 添加換行符
        arduino.write(message.encode())
        print(f"已發送數據: {value}")
        time.sleep(1)  # 可以根據需要調整等待時間

def read_response():
    while True:
        if arduino.in_waiting:  # 檢查是否有可用的數據
            response = arduino.readline().decode('utf-8', errors='ignore').strip()
            if response:
                print(f"Arduino 回應: {response}")

# 創建並啟動兩個執行緒
thread_send = threading.Thread(target=send_value)
thread_read = threading.Thread(target=read_response)

thread_send.start()
thread_read.start()

# 等待兩個執行緒結束
thread_send.join()
thread_read.join()