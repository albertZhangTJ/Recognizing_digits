#include "neuron.h"
#include <vector>
#include <chrono>
#include <random>
#include <math.h>
#include <iostream>
#include <exception>

using namespace std;
double lambda=0.01;

//the logistic function
double sigmoid(double inp){
    return 1.0/1+pow(M_E, -inp);
}
double d_sigmoid(double z){
    return pow(M_E, z)/pow(1+pow(M_E, z), 2);
}
//not the actual dot product in mathematics, but instead with a constant value in w
double dot_product_mod(const vector<double>& w, const vector<double>& x){
    double ans=0;
    if (w.size()!=(x.size()+1)){
        cout<<"\033[1;31mERROR: incompatible dot product size, weight has "<<w.size()<<", input has "<<x.size()<<"\033[0m"<<endl;
        throw new exception();
    }
    for (int i=0; i<x.size(); i++){
        ans+=w[i]*x[i];
    }
    //bias
    ans=ans+w[w.size()-1];
    return ans;
}

neuron::neuron(int num_input){
    this->inp_size=num_input;
    if (num_input==1){
        this->is_input_neuron=true;
        this->weights.push_back(1);
        this->weights.push_back(0); //input neurons have bias of 0
    }
    else {
        this->is_input_neuron=false;
        mt19937 mt(chrono::system_clock::now().time_since_epoch().count());
        //one extra for the bias
        for (int i=0; i<=this->inp_size; i++){
            this->weights.push_back(mt()%100/101.0);
        }
    }
}

double neuron::generate_inference(const vector<double>& inps){
    if (inps.size()!=this->inp_size){
        cout<<"\033[1;31mERROR: input vector size "<<inps.size()<<" incompatible with neuron size "+this->inp_size<<"\033[0m"<<endl;
        throw new exception();
    }
    return sigmoid(dot_product_mod(this->weights, inps));
}

vector<double> neuron::update(const std::vector<double>& inps, const vector<double>& errs){
    vector<double> ans;
    if (this->is_input_neuron){
        return ans;
    }

    if (inps.size()!=this->inp_size){
        cout<<"\033[1;31mERROR: Back propagating: input vector size "<<inps.size()<<" incompatible with neuron size "<<this->inp_size<<"\033[0m"<<endl;
        throw new exception();
    }

    double avg_err=0;
    for (int i=0; i<errs.size(); i++){
        avg_err+=errs[i];
    }
    avg_err/=errs.size();
    
    double d_sig=d_sigmoid(dot_product_mod(this->weights, inps));

    //how the current neuron thinks the previous activations should change
    for (int i=0; i<inps.size(); i++){
        ans.push_back(d_sig * this->weights[i] * avg_err);
    }

    //changing the weights of the current neuron
    for (int i=0; i<inps.size(); i++){
        this->weights[i]+=d_sig * inps[i] * avg_err * lambda;
    }

    //changing the bias of the current neuron
    this->weights[this->weights.size()-1]+=d_sig * avg_err * lambda;

    return ans;
}