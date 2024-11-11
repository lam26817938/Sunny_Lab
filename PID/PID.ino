const int MAX_HISTORY_SIZE = 50;  // Adjust to store enough data points within 3 seconds at the PID_INTERVAL rate
int pwmHistory_A[MAX_HISTORY_SIZE];      
int speedHistory_A[MAX_HISTORY_SIZE];     
unsigned long timeHistory_A[MAX_HISTORY_SIZE];
int historyIndex_A = 0;
const unsigned long SPEED_STABILITY_TIME = 5000; // 3 seconds in milliseconds

// L298N Motor Driver Pin Assignments
const int IN1_A = 7;
const int IN2_A = 10;
const int ENA_A = 11;  // PWM Pin

// L298N Motor Driver Pin Assignments for Motor B
const int IN1_B = 5;
const int IN2_B = 6;
const int ENB_B = 9;

// Encoder Pins for Motor A
const int encoderPinA_A = 2; 
const int encoderPinB_A = 4;

// Encoder Pins for Motor B
const int encoderPinA_B = 3;
const int encoderPinB_B = 8;

// Encoder and PID variables for Motor A
volatile long encoderCount_A = 0;
long prevEncoderCount_A = 0;
float currentSpeed_A = 0;
float targetSpeed_A = 900.0;
float Kp_A = 0.2, Ki_A = 0.05, Kd_A = 0.01;
float integral_A = constrain(integral_A, -100, 100), lastError_A = 0;

// PID calculation interval (in milliseconds)
const int PID_INTERVAL = 100;

//unsigned long startTime = 0;
//unsigned long speedChangeInterval = 60000;

void encoderISR_A() {
  if (digitalRead(encoderPinA_A) == digitalRead(encoderPinB_A)) {
    encoderCount_A++;
    //Serial.println(encoderCount_A);
  } else {
    encoderCount_A--;
    //Serial.print(encoderCount_A);
  }
}

void setup() {
  // Motor A control pins setup
  pinMode(IN1_A, OUTPUT);
  pinMode(IN2_A, OUTPUT);
  pinMode(ENA_A, OUTPUT);
  
  // Encoder pins setup
  pinMode(encoderPinA_A, INPUT);
  pinMode(encoderPinB_A, INPUT);
  
  // Enable interrupts for encoders
  attachInterrupt(digitalPinToInterrupt(encoderPinA_A), encoderISR_A, CHANGE);
  
  // Initialize serial for debugging
  Serial.begin(9600);
  

}
void loop() {
  // PID control for Motor A
    if (Serial.available() > 0) {
      String receivedData = Serial.readStringUntil('\n');  // 讀取數據直到換行符
      if (receivedData.startsWith("C:")) {  // 檢查數據是否來自電腦
          String valueString = receivedData.substring(2);  // 去掉標識符 'C:'
          int value = valueString.toInt();  // 將數據轉換成整數
          Serial.print("Received from computer: ");
          Serial.println(value);  // 顯示數據
          // 在這裡執行其他操作
      }
  }
  handlePIDMotorA();
  // PID control for Motor B
  //handlePIDMotorB();
}

void handlePIDMotorA() {
  static unsigned long prevTime_A = millis();
  unsigned long currentTime_A = millis();
  unsigned long deltaTime_A = currentTime_A - prevTime_A;

  if (deltaTime_A >= PID_INTERVAL) {
    prevTime_A = currentTime_A;

    // Calculate speed from encoder counts
    long deltaCount_A = encoderCount_A - prevEncoderCount_A;
    if (deltaTime_A > 0 && deltaCount_A != 0) {
      currentSpeed_A = (deltaCount_A / (float)deltaTime_A) * 1000.0;
    } else {
      currentSpeed_A = 0;
    }

    // PID calculations
    float error_A = targetSpeed_A - currentSpeed_A;
    integral_A += error_A * (deltaTime_A / 1000.0);
    float derivative_A = (error_A - lastError_A) / (deltaTime_A / 1000.0);
    float output_A = (Kp_A * error_A) + (Ki_A * integral_A) + (Kd_A * derivative_A);
    int motorPWM_A = constrain(output_A, 50, 250);
    setMotorA(1, motorPWM_A);

    // Check if speed is within 8% range
    float lowerBound = targetSpeed_A * 0.95;
    float upperBound = targetSpeed_A * 1.05;
    bool speedWithinRange = (currentSpeed_A >= lowerBound && currentSpeed_A <= upperBound);

    // Store current PWM and timestamp in circular buffer
    pwmHistory_A[historyIndex_A] = motorPWM_A;
    timeHistory_A[historyIndex_A] = currentTime_A;
    speedHistory_A[historyIndex_A] = currentSpeed_A;
    historyIndex_A = (historyIndex_A + 1) % MAX_HISTORY_SIZE;

    // Calculate average PWM if speed has been stable for past 5 seconds
    unsigned long oldestTime = currentTime_A - SPEED_STABILITY_TIME;
    float sumPWM = 0;
    float sumSpeed = 0;
    int count = 0;
    for (int i = 0; i < MAX_HISTORY_SIZE; i++) {
      if (timeHistory_A[i] >= oldestTime) {
        sumPWM += pwmHistory_A[i];
        sumSpeed += speedHistory_A[i];
        count++;
      }
    }
    int averagePWM = (count > 0) ? (sumPWM / count) : motorPWM_A;
    int averageSpeed = (count > 0) ? (sumSpeed / count) : 5000;

    // if (speedWithinRange && count > 1) {
    //   setMotorA(1, averagePWM);
    //   Serial.print("Motor A is stable. Setting PWM to average: ");
    //   Serial.print(averagePWM);
    //   Serial.print(" | Stable motor speed: ");
    //   Serial.println(averageSpeed);
    // } else {
    //   Serial.print("Motor A | Target Speed: ");
    //   Serial.print(targetSpeed_A);
    //   Serial.print(" | Current Speed: ");
    //   Serial.print(currentSpeed_A);
    //   Serial.print(" | PWM Output: ");
    //   Serial.println(motorPWM_A);
    // }

    lastError_A = error_A;
    prevEncoderCount_A = encoderCount_A;
  }
}

void setMotorA(int direction, int speed) {
  if (direction == 1) {  // Forward
    digitalWrite(IN1_A, HIGH);
    digitalWrite(IN2_A, LOW);
  } else {  // Stop
    digitalWrite(IN1_A, LOW);
    digitalWrite(IN2_A, LOW);
  }
  analogWrite(ENA_A, speed);
}
