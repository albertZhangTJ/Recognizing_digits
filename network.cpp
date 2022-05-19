#include "network.h"
#include "neuron.h"
#include "file_parser.h"
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;


//打印一个vector，调试用
void print(vector<double>& v){
    for (double d: v){
        cout<<d<<" ";
    }
    cout<<endl;
}

//损失函数（不过没用到）
double loss(vector<double> outputs, int label){
    double ans=0;
    for (int i=0; i<outputs.size(); i++){
        if (i==label){
            ans+=pow((1-outputs[i]), 2);
        }
        else {
            ans+=pow(outputs[i], 2);
        }
    }
    return ans;
}


//构造函数
//从输入层开始，每一层的neuron数量是上一层的一半
network::network(int inp_size, int oup_size){
    int num=inp_size;
    int last_layer=1;
    while (num>oup_size){
        vector<neuron> to_push;
        for (int i=0; i<num; i++){
            neuron to_add(last_layer);
            to_push.push_back(to_add);
        }
        this->neurons.push_back(to_push);
        last_layer=num;
        num/=2;
    }
    vector<neuron> oup_layer;
    for (int i=0; i<oup_size; i++){
        neuron to_add(last_layer);
        oup_layer.push_back(to_add);
    }
    this->neurons.push_back(oup_layer);
}


vector<double> network::forward_propagate(vector<char> images){
    vector<double> last_layer_output;
    vector<double> cur_layer_output;
    this->inputs.clear();
    //feeding raw input through the input neurons
    for (int i=0; i<images.size(); i++){
        //since the raw value of pixels are ranged 0-255
        //to avoid everything being clustered near 1 after applied through the sigmoid function
        //recenter the distribution to 0
        vector<double> tmp={int(images[i])*1.0 - 127}; 
        last_layer_output.push_back(this->neurons[0][i].generate_inference(tmp));
    }

    //passing the outputs of the input neurons (which are centered raw inputs through the sigmoid function)
    for (int i=1; i<this->neurons.size(); i++){
        for (int j=0; j<this->neurons[i].size(); j++){
            cur_layer_output.push_back(this->neurons[i][j].generate_inference(last_layer_output));
        }
        this->inputs.push_back(last_layer_output);
        last_layer_output=cur_layer_output;
        cur_layer_output.clear();
    }
    //print(last_layer_output);
    return last_layer_output;
}


void network::back_propagate(vector<double> outputs, int label){
    //cout<<label<<endl;
    //右边一层生成的的error
    vector<vector<double>> last_err;
    //本层生成的来自左边一层输入的error
    vector<vector<double>> cur_err;

    //从最右边开始
    //每个输出层neuron的error
    for (int i=0; i<outputs.size(); i++){
        if (i==label){
            last_err.push_back({1-outputs[i]});
        }
        else {
            last_err.push_back({0-outputs[i]});
        }
    }

    //反向传播
    for (int i=this->neurons.size()-1; i>0; i--){
        cur_err.clear();
        for (int j=0; j<neurons[i].size(); j++){
            //cout<<this->inputs.size()<<" "<<i<<" "<<this->inputs[i].size()<<endl;

            //让一个neuron调整自己的weights和bias
            //同时生成这个neuron认为的上一层activation应该怎么变化
            vector<double> gen=neurons[i][j].update(this->inputs[i-1],last_err[j]);
            //本层第一个neuron的时候需要生成上一层每一个neuron对应的vector
            if (cur_err.empty()){
                for (double err: gen){
                    cur_err.push_back({err});
                }
            }
            //否则只需要把生成的error放到对应的vector里
            else {
                for (int k=0; k<gen.size(); k++){
                    cur_err[k].push_back(gen[k]);
                }
            }
        }
        last_err=cur_err;
    }
}

void network::train(idx3 images, idx1 labels){
    if (images.size!=labels.size){
        cout<<"\033[1;31mERROR: images set and labels set does not match"<<"\033[0m"<<endl;
        throw new exception;
    }
    for (int i=0; i<images.images.size(); i++){
        this->back_propagate(forward_propagate(images.images[i]), labels.labels[i]);
        if ((i+1)%(images.size/100)==0){
            cout<<"Training, "<<i+1<<" images used"<<endl;
        }
    }
}

//根据forward_propagation的输出，选出最接近1的那个
//返回index
int network::recognize(std::vector<char> image){
    vector<double> outputs=this->forward_propagate(image);
    double max=-1;
    int max_ind=0;
    for (int i=0; i<10; i++){
        //cout<<outputs[i]<<" ";
        if (outputs[i]>max){
            max=outputs[i];
            max_ind=i;
        }
    }
    //cout<<endl;
    return max_ind;
}

void network::test(idx3 images, idx1 labels){
    if (images.size!=labels.size){
        cout<<"\033[1;31mERROR: images set and labels set does not match"<<"\033[0m"<<endl;
        throw new exception;
    }
    int total=0;
    int err=0;
    for (int i=0; i<images.images.size(); i++){
        if (this->recognize(images.images[i])!=labels.labels[i]){
            err++;
        }
        total++;
        if ((i+1)%(images.size/100)==0){
            cout<<"Testing, "<<i+1<<" images used"<<endl;
            cout<<"Current accuracy: "<<100-err*100/total<<"%"<<endl<<endl;
        }
    }
}