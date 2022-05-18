#ifndef NEURON_H
#define NEURON_H
#include <vector>

double sigmoid(double inp);

class neuron{
    public:
        neuron(int num_input);
        std::vector<double> update(const std::vector<double>& inps, const std::vector<double>& errs);
        double generate_inference(const std::vector<double>& inps);
    private:
        int inp_size;
        bool is_input_neuron;
        std::vector<double> weights;
};

#endif //NEURON_H