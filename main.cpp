#include <iostream>
#include <chrono>

#include <torch/script.h>

constexpr int NUM_OF_LOOP = 1000;

int main(int argc, char *argv[]){
    using clock = std::chrono::high_resolution_clock;

    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";

    module.eval();
    module.to(c10::DeviceType::CUDA);

    std::vector<torch::jit::IValue> inputs = {
        torch::ones({1, 3, 244, 244}).to(c10::DeviceType::CUDA)
    };

    // warmup
    for(int i = 0; i < 10; ++i) module.forward(inputs);

    {
        double sum_duration = 0.0;
        for(int i = 0; i < NUM_OF_LOOP; ++i){
            auto start = clock::now();
            at::Tensor output = module.forward(inputs).toTensor();
            auto end = clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            sum_duration += duration;
        }
        std::cout << sum_duration / NUM_OF_LOOP * 1000. << "ms/frame, " << NUM_OF_LOOP / sum_duration << "fps" << std::endl;
    }

    ///
    {
        double sum_duration = 0.0;
        for(int i = 0; i < NUM_OF_LOOP; ++i){
            auto start = clock::now();
            module.forward(inputs);
            auto end = clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            sum_duration += duration;
        }
        std::cout << sum_duration / NUM_OF_LOOP * 1000. << "ms/frame, " << NUM_OF_LOOP / sum_duration << "fps" << std::endl;
    }

    {
        double sum_duration = 0.0;
        for(int i = 0; i < NUM_OF_LOOP; ++i){
            auto start = clock::now();
            at::Tensor output = module.forward(inputs).toTensor();
            auto end = clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            sum_duration += duration;
        }
        std::cout << sum_duration / NUM_OF_LOOP * 1000. << "ms/frame, " << NUM_OF_LOOP / sum_duration << "fps" << std::endl;
    }

}
