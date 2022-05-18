#ifndef NETWORK_H
#define NETWORK_H
#include "neuron.h"
#include "file_parser.h"
#include <vector>

class network{
    public:
        network(int inp_size, int oup_size);
        std::vector<double> forward_propagate(std::vector<char> image);
        void back_propagate(std::vector<double> outputs, int label);
        void train(idx3 images, idx1 labels);
        void test(idx3 images, idx1 labels);
        int recognize(std::vector<char> image);
    private:
        std::vector<std::vector<neuron>> neurons;
        std::vector<std::vector<double>> inputs;
        int inp_size;
        int oup_size;
};

#endif //NETWORK_H
