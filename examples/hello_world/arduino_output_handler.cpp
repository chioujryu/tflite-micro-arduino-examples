/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>

#include "Arduino.h"
#include "constants.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

// The pin of the Arduino's built-in LED
int led = LED_BUILTIN;   ////這是arduino本機亮燈的寫法

// Track whether the function has run at least once
//追蹤函數是否已經運行過至少一次。
bool initialized = false;

// Animates a dot across the screen to represent the current x and y values
//將一個點在屏幕上移動，以表示當前的 x 和 y 值。
void HandleOutput(float x_value, float y_value) {
  // Do this only once
  if (!initialized) {
    // Set the LED pin to output
    pinMode(led, OUTPUT);
    initialized = true;
  }

  // Calculate the brightness of the LED such that y=-1 is fully off
  // and y=1 is fully on. The LED's brightness can range from 0-255.
  //計算 LED 的亮度，使得當 y=-1 時完全關閉，而當 y=1 時完全開啟。LED 的亮度範圍為 0-255。  
  int brightness = (int)(127.5f * (y_value + 1));

  // The y value is not actually constrained to the range [-1, 1], so we need to
  // clamp the brightness value before sending it to the PWM/LED.
  int brightness_clamped = std::min(255, std::max(0, brightness));

  // Set the brightness of the LED. If the specified pin does not support PWM,
  // this will result in the LED being on when brightness_clamped > 127, off
  // otherwise.
  analogWrite(led, brightness_clamped);

  // Log the current brightness value for display in the Arduino plotter
  MicroPrintf("%d\n", brightness);
  delay(33); 
}
