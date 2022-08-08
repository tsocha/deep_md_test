#include "paddle/include/paddle_inference_api.h"
#include <chrono>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <numeric>

DEFINE_string(dirname, "model.ckpt", "Directory of the inference model.");

namespace paddle_infer {

using Time = decltype(std::chrono::high_resolution_clock::now());
Time time() { return std::chrono::high_resolution_clock::now(); };
double time_diff(Time t1, Time t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

void PrepareTRTConfig(Config *config) {
  config->SetModel(FLAGS_dirname + "/model.pdmodel", FLAGS_dirname + "/model.pdiparams");
  config->EnableUseGpu(100, 0);
  //Uncomment for CPU Backend
  //config->DisableGpu();
}

bool test_map_cnn(int batch_size, int repeat) {
  Config config;
  PrepareTRTConfig(&config);
  auto predictor = CreatePredictor(config);

  int channels = 3;
  int height = 224;
  int width = 224;
  int input_num = channels * height * width * batch_size;

  // prepare inputs
  std::vector<float> input(input_num, 0);
  auto input_names = predictor->GetInputNames();
  std::cout << "name: " << input_names[0] << std::endl;
  auto input_t = predictor->GetInputHandle(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->CopyFromCpu(input.data());

  // run
  auto time1 = time();
  for (size_t i = 0; i < repeat; i++) {
    CHECK(predictor->Run());
  }
  auto time2 = time();
  std::cout << "batch: " << batch_size << " predict cost: "
            << time_diff(time1, time2) / static_cast<float>(repeat) << "ms"
            << std::endl;

  // get the output
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  for (size_t j = 0; j < out_num; j += 100) {
    LOG(INFO) << "output[" << j << "]: " << out_data[j];
  }
  return true;
}

template<typename T>
void ReadFromBinary(const std::string& path,
                    int& rank,
                    std::vector<int>& shape,
                    std::vector<T>& data) {
  // open the file:
  float f;
  std::ifstream file(path, std::ios::binary);

  file.read(reinterpret_cast<char*>(&f), sizeof(float));
  rank = static_cast<int>(f);
  
  for(size_t i=0; i<rank; i++) {
    file.read(reinterpret_cast<char*>( &f ), sizeof(float));
    shape.push_back(static_cast<int>(f));
  }

  while (file.read(reinterpret_cast<char*>(&f), sizeof(float))) {
    data.push_back(static_cast<T>(f));
  }
}

template<typename T>
void ConfigureInputs(const std::string& filename,
                     const std::string& input_name,
                     const std::shared_ptr<Predictor>& predictor) {
    // Read Input
    int rank;
    std::vector<int> shape;
    std::vector<T> data;
    ReadFromBinary<T>(filename, rank, shape, data);
    
    // Set Input
    auto tensor = predictor->GetInputHandle(input_name);
    tensor->Reshape(shape);
    tensor->CopyFromCpu(data.data());
}

bool test_dp_infer(void) {
  Config config;
  PrepareTRTConfig(&config);
  auto predictor = CreatePredictor(config);
  // prepare inputs
  auto input_names = predictor->GetInputNames();
  // coord_ 
  ConfigureInputs<float>("data_convert/coord.bin", input_names[0], predictor);
  ConfigureInputs<int>("data_convert/type.bin", input_names[1], predictor);
  ConfigureInputs<int>("data_convert/natoms_vec.bin", input_names[2], predictor);
  ConfigureInputs<float>("data_convert/box.bin", input_names[3], predictor);
  ConfigureInputs<int>("data_convert/default_mesh.bin", input_names[4], predictor);

  predictor->Run();
  
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  for (size_t j = 0; j < out_num; j += 100) {
    LOG(INFO) << "output[" << j << "]: " << out_data[j];
  }

}
} // namespace paddle_infer

int main(int argc, char *argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  paddle_infer::test_dp_infer();
  return 0;
}
