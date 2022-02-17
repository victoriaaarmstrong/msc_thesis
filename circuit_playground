#include <Adafruit_TinyUSB.h>

// Imports necessary board packages
#include <Adafruit_CircuitPlayground.h>
#include <Adafruit_Circuit_Playground.h>

// Necessary for the use of airpumps
#include <Arduino.h>

// Libraries for heart rate monitor
#include <Wire.h>
#include <MAX30105.h>
#include "heartRate.h"

// Constants for heart rate collection
const byte RATE_SIZE = 4; //Increase this for more averaging. 4 is good.
byte rates[RATE_SIZE]; //Array of heart rates
byte rateSpot = 0;
long lastBeat = 0; //Time at which the last beat occurred
float beatsPerMinute;
int beatAvg;
MAX30105 particleSensor;

// Pump variables
int pump = A1;
int vacuum = A2;
int current_left_state;
int current_right_state;
int last_right_state;
int last_left_state;
int pump_state = LOW;
int vacuum_state = LOW;

// Variable instantiations for writing to CSV file
float temp = 0;
float ir = 0;
float bpm = 0;
float accX = 0;
float accY = 0;
float accZ = 0;
float user_act = 0;
float pump_on = 0.0;
float vacuum_on = 0.0;


void setup() {
  CircuitPlayground.begin();
  Serial.begin(115200);
  
  Serial.println("Initializing...");

  pinMode(pump, OUTPUT);
  pinMode(vacuum, OUTPUT);

  // Initialize and set up heart rate sensor
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) //Use default I2C port, 400kHz speed
  {
    Serial.println("MAX30105 was not found. Please check wiring/power. ");
    while (1);
  }
  
  // Setup to sense a nice looking saw tooth on the plotter
  byte ledBrightness = 0x1F; //Options: 0=Off to 255=50mA
  byte sampleAverage = 8; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 3; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  int sampleRate = 100; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200 
  int pulseWidth = 411; //Options: 69, 118, 215, 411
  int adcRange = 2048; //Options: 2048, 4096, 8192, 16384

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings
  particleSensor.setPulseAmplitudeRed(0x0a);
  particleSensor.setPulseAmplitudeGreen(0);

  current_left_state = CircuitPlayground.leftButton();
  current_right_state = CircuitPlayground.rightButton();
} 

void loop() {
  last_right_state = current_right_state;
  last_left_state = current_left_state;

  current_right_state = CircuitPlayground.rightButton();
  current_left_state = CircuitPlayground.leftButton();
  
  if(last_left_state == HIGH && current_left_state == LOW) {
    pump_state = !pump_state;
    digitalWrite(pump, pump_state); 
  }

  if(last_right_state == HIGH && current_right_state == LOW) {
    vacuum_state = !vacuum_state;
    digitalWrite(vacuum, vacuum_state); 
  }


  // Read from the sensor
  long irValue = particleSensor.getIR();

  
  if (checkForBeat(irValue) == true)
  {
    //We sensed a beat!
    long delta = millis() - lastBeat;
    lastBeat = millis();

    beatsPerMinute = 60 / (delta / 1000.0);
  }

  // Set the values of the data we want to export to CSV
  ir = irValue; //rename in the file
  bpm = beatsPerMinute;
  temp = CircuitPlayground.temperature();
  accX = CircuitPlayground.motionX();
  accY = CircuitPlayground.motionY();
  accZ = CircuitPlayground.motionZ();

  if (pump_state == LOW) {
    pump_on = 0.0;
  } else if (pump_state == HIGH) {
    pump_on = 1.0;
  }

  if (vacuum_state == LOW) {
    vacuum_on = 0.0;
  } else if (vacuum_state == HIGH) {
    vacuum_on = 1.0;
  }
  
  
  // Write variables to a string that can be read through the serial port using processing
  String string1 = ",";
  String string2 = ir + string1 + bpm + string1 + pump_on + string1 + vacuum_on + string1 + temp + string1 + accX + string1 + accY + string1 + accZ;
  Serial.println(string2);
  delay(100); //do I need delay - play with this
}
