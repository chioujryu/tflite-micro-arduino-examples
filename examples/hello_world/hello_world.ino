/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================

training：

==============================================================================*/

#include <TensorFlowLite.h>

//1. 要使用 TensorFlow Lite for Microcontrollers 库，我们必须包含以下头文件：
#include "tensorflow/lite/micro/all_ops_resolver.h" // 提供解釋器用來運行模型的操作。 (https://oreil.ly/O0qgy)
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h" //输出调试信息。
#include "tensorflow/lite/micro/micro_interpreter.h"  // 包含加載和運行模型的代碼。
#include "tensorflow/lite/schema/schema_generated.h"  //包含 TensorFlow Lite FlatBuffer 模型文件架构的模式。
#include "tensorflow/lite/version.h" //提供 Tensorflow Lite 架构的版本控制信息。

//2. 包含模型头文件
#include "model.h"

//3. 其餘標頭檔
#include "constants.h" // 定義常用常數
#include "main_functions.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"


// 4. 宣告命名空間，方便後續程式呼叫使用
// 這些變數都被包在namespace裡面，這代表雖然他們可以在這個檔案內的任何地方存取
//但無法再專案的任何其他檔案裡面存取。
//這種做法有助於防止兩個不同的檔案剛好定義名稱相同的變數，進而造成問題。
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr; // 模型指標
tflite::MicroInterpreter* interpreter = nullptr; // 直譯器指標
TfLiteTensor* input = nullptr; // 輸人指標
TfLiteTensor* output = nullptr; // 輸出指標
int inference_count = 0; // 推論計數器

//  5. 建立一個記憶體區域，供輸入，輸出與中間陣列使用
//  你可能要透過試誤法來找出模型的最小值
constexpr int kTensorArenaSize = 2048;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace


// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // 設定logging
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


  // 6. 加载模型
  //下面的代码使用了 model.h 中声明的 char 数组和 g_model 中的数据实例化模型。
  //然后，我们检查模型，以确保它的架构版本与我们正在使用的版本兼容：
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // 6. 實例化運算解析器
  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  // 聲明了一個 AllOpsResolver 實例。 解釋器將使用它來訪問模型所使用的運算：
  static tflite::AllOpsResolver resolver;
  // AllOpsResolver 會載入 TensorFlow Lite for Microcontrollers 中可用的所有運算，
  // 而這些運算會佔用大量記憶體。 由於給定的模型僅會用到這些運算中的一部分，
  // 因此建議在實際應用中僅載入所需的運算。

  // 額外補充：
  // 这是使用另一个类 MicroMutableOpResolver 来实现的。
  // 您可以在 Micro speech 示例的 micro_speech_test.cc 中了解如何使用它。
  //https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/micro_speech/micro_speech_test.cc



  //7. 建構執行模型的解譯器
  // Build an interpreter to run the model with.
  // 宣告`tensor_arena之後，我們可以建構執行模型的解譯器。這是它的樣子
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  // 8. 使用tensor_arena為模型的張量配置記憶體
  // 在之前，我們設定了tensor_arena陣列來留一塊記憶體，AllocateTensor()方法會遍歷模型定義的所有張量
  // 並且將tensor_arena的記憶體指派給他們
  TfLiteStatus allocate_status = interpreter->AllocateTensors();  
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // 取得指向模型輸入與輸出張量的指標
  // 建立解譯器之後，我們要提供輸入給模型，所以將輸入資料寫至模型的輸入張量：
  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);    // 取得指向模型輸入張量的指標
  output = interpreter->output(0);  // 取得指向模型輸出張量的指標
  // 我們要呼叫解譯器的input()方法來取得指向輸入張量的指標。因為模型可能有多個輸入張量，
  // 我們要對input()傳入一個索引，來指定想要的張量。這個例子的模型只有一個輸入張量，所以它的索引是0
  // 而output也只有一個輸出張量，所以引數也是0。

  //你可能會，我們為什麼可以在執行推斷之前跟輸出互動？因為TfLiteTensor是一個擁有data成員的struct，
  //data指向一塊已經配置好，用來儲存輸出的記憶體區域，雖然這個struct的data還沒有被填入資料，
  //但他們依然是存在的，，，，，，，，


  // Keep track of how many inferences we have performed.
  // 初始化推論計數器
  // 推論計數器可以計算AI MODEL推論了幾次
  inference_count = 0;
}

// The name of this function is important for Arduino compatibility.
void loop() {


  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  //計算一個要餵入模型的 x 值。我們將當前的推論計數器與每個週期的推論數量進行比較，
  //以確定我們在模型訓練時可能的 x 值範圍內的位置，並使用此位置來計算一個值。
  float position = static_cast<float>(inference_count) /     //inference_count / kInferencesPerCycle = position
                   static_cast<float>(kInferencesPerCycle);   //kInferencesPerCycle in the arduino_constants.cpp /
                                                            //目前設定為200, 大概等於6.6秒
  
  //定義輸入的X值
  float x = position * kXrange;  
  // (0/200) * 2PI
  // x = 0 ~ 2PI

  // Quantize the input from floating-point to integer
  // 將輸入從浮點數量化為整數
  int8_t x_quantized = (x / input->params.scale) + input->params.zero_point; 
  //(input->params.scale) = 0.02左右
  //(input->params.zero_point) = -128左右

  //可以運行看看這兩行，但要把`arduino_output_handler.cpp`的`MicroPrintf("%d\n", brightness);`給註解掉
  //Serial.print("x_quantized："); 
  //Serial.println(x_quantized);


  // Place the quantized input in the model's input tensor
  //提供一個輸入值
  //將整數放入模型的輸入張量中
  input->data.int8[0] = x_quantized;  //假如要接收浮點數的話，要寫成data.f[0] = 0.;  數字寫成0. 是為了讓數字為浮點數

  //補充：
  //data變數是個TfLitePtrUnion，它是個union，這是一種特殊的C++資料型態，
  //可讓你在記憶體的同一個位置儲存不同的資料型態。因為收到的張量可能存有多種
  //不同的資料型態（例如浮點數、整數或布林），union型態可以協助我們完美地儲存它。
  //TfLitePtrUnion union是在https://oreil.ly/v4h7K。裡面宣告的


  // 用這個來執行模型，並確定它成功了
  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke(); //Invoke() 調用的意思
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed on x: %f\n", static_cast<double>(x));
    return;
  }
  //當我們呼叫interpreter的invoke()時，Tensorflow Lite解譯器就會執行模型推論，
  //並將輸入資料換成輸出，輸出會被存放在模型的輸出張量。

 

  // Obtain the quantized output from model's output tensor
  // 從模型的輸出張量中獲取量化輸出
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  // 將從整數轉換為浮點數的輸出進行反量化。
  float y = (y_quantized - output->params.zero_point) * output->params.scale;  //output->params.scale用於反量化使用


  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  //增加推論計數器的值，並在每個週期達到總數時將其重置。
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
