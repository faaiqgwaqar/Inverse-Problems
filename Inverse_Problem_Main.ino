#include <Adafruit_MAX31865.h>

// Use software SPI: CS, DI, DO, CLK
Adafruit_MAX31865 thermo = Adafruit_MAX31865(10, 11, 12, 13);
// Use hardware SPI, just pass in the CS pin
//* Adafruit_MAX31865 thermo = Adafruit_MAX31865(10);*

// The value of the Rref resistor.
#define RREF      4300.0
// The 'nominal' 0-degrees-C resistance of the sensor
#define RNOMINAL  1000.0

unsigned long CurrentTime;
unsigned long StartTime;
bool start_clock;

void setup() {
  // Define the baud rate(UART)
  Serial.begin(115200);

  // Initialize 3 wire interfacing RTD sensor
  thermo.begin(MAX31865_3WIRE);

  start_clock = true;

  Serial.println("Time [ms] \t Temp [C]");
}


void loop() {
  uint16_t rtd = thermo.readRTD();

  // Serial.print("RTD value: "); Serial.println(rtd);
  float ratio = rtd;
  ratio /= 32768;
  /* Serial.print("Ratio = "); Serial.println(ratio,8);
  Serial.print("Resistance = "); Serial.println(RREF*ratio,8);
  Serial.print("Temperature = ");*/
  if(start_clock) 
  {
    StartTime = millis();
    CurrentTime = StartTime;
    start_clock = false;
  }
  else CurrentTime = millis();

  Serial.print(CurrentTime - StartTime);
  Serial.print("\t");
  Serial.print(thermo.temperature(RNOMINAL, RREF));
  Serial.print("\n");

  // Check and print any faults
  uint8_t fault = thermo.readFault();
  if (fault) {
    Serial.print("Fault 0x"); Serial.println(fault, HEX);
    if (fault & MAX31865_FAULT_HIGHTHRESH) {
      Serial.println("RTD High Threshold"); 
    }
    if (fault & MAX31865_FAULT_LOWTHRESH) {
      Serial.println("RTD Low Threshold"); 
    }
    if (fault & MAX31865_FAULT_REFINLOW) {
      Serial.println("REFIN- > 0.85 x Bias"); 
    }
    if (fault & MAX31865_FAULT_REFINHIGH) {
      Serial.println("REFIN- < 0.85 x Bias - FORCE- open"); 
    }
    if (fault & MAX31865_FAULT_RTDINLOW) {
      Serial.println("RTDIN- < 0.85 x Bias - FORCE- open"); 
    }
    if (fault & MAX31865_FAULT_OVUV) {
      Serial.println("Under/Over voltage"); 
    }
    thermo.clearFault();
  }

  for(int i = 0; i < 10; i++) delay(1000);
}
