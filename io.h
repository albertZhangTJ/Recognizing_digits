//since the functions defined in this header is really simple
//and cuz I'm lazy
//I directly implemented everything in this header file
#ifndef IO_H
#define IO_H
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iterator>
//reading a binary file
std::vector<char> read_file(std::string file_name){
    std::cout<<"Reading binary file: "<<file_name<<" ..."<<std::flush;
    std::ifstream input(file_name, std::ios::binary );
    std::vector<char> buffer(std::istreambuf_iterator<char>(input), {});
    std::cout<<" \033[1;32mDONE\033[0m"<<std::endl;
    return buffer;
}
#endif //IO_H