/*
  Ball-on-Plate – receptor de comandos para 2 servos
  -------------------------------------------------
  Protocolo serie (115200 baud):
    S,<alpha_deg>,<beta_deg>\n
  Ejemplo:
    S,3.50,-1.20\n
  Conexiones típicas:
    Servo X (alpha) → pin 9
    Servo Y (beta)  → pin 10
*/

#include <Servo.h>

Servo servoX;
Servo servoY;

const int PIN_X = 9;
const int PIN_Y = 10;

// Ángulo central de los servos (ajustar según tu mecánica)
const float CENTER_X = 90.0;
const float CENTER_Y = 90.0;

// Límite de inclinación de la placa (debe coincidir con el Python)
const float MAX_ANGLE = 12.0;

void setup() {
  Serial.begin(115200);
  servoX.attach(PIN_X);
  servoY.attach(PIN_Y);

  // Posición horizontal al arrancar
  servoX.write(CENTER_X);
  servoY.write(CENTER_Y);

  Serial.println("Ball-on-Plate servos listos");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    if (line.startsWith("S,")) {
      // Formato: S,alpha,beta
      int c1 = line.indexOf(',');
      int c2 = line.indexOf(',', c1 + 1);

      if (c1 > 0 && c2 > c1) {
        float alpha = line.substring(c1 + 1, c2).toFloat();
        float beta  = line.substring(c2 + 1).toFloat();

        // Saturar
        alpha = constrain(alpha, -MAX_ANGLE, MAX_ANGLE);
        beta  = constrain(beta,  -MAX_ANGLE, MAX_ANGLE);

        // Convertir inclinación de placa → ángulo de servo
        // (ajusta el signo según cómo hayas montado los servos)
        float cmdX = CENTER_X + alpha;
        float cmdY = CENTER_Y + beta;

        servoX.write(cmdX);
        servoY.write(cmdY);
      }
    }
  }
}
