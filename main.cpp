//All the runtime exceptions generated are deliberately not handled
//since they indicate errors in either the model or the input file
//so the model should be stopped when an exception occurs
#include "io.h"
#include "file_parser.h"
#include "neuron.h"
#include "network.h"
#include <iostream>
#include <exception>

using namespace std;

int main(){
    try {
        idx3 train_images=parse_idx3(read_file("train-images.idx3-ubyte"));
        idx1 train_labels=parse_idx1(read_file("train-labels.idx1-ubyte"));
        network net(train_images.rows*train_images.cols, 10);
        net.train(train_images, train_labels);
    }
    catch (exception e){
        cout<<"Program terminated"<<endl;
    }
    return 0;
}