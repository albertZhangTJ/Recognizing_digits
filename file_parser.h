//since the functions defined in this header is relatively simple
//and cuz I'm lazy
//I directly implemented everything in this header file

#ifndef FILE_PARSER_H
#define FILE_PARSER_H
#include <vector>
#include <iostream>
#include <exception>
struct idx3{
    int rows,cols;
    int size; //number of images
    std::vector<std::vector<char>> images; //each image is represented as a single vector
};

struct idx1{
    int size; //number of images
    std::vector<int> labels;
};

//left -> right: MSB -> LSB
inline int chars_to_int(char c1, char c2, char c3, char c4){
    return int((unsigned char)(c1) << 24 |
            (unsigned char)(c2) << 16 |
            (unsigned char)(c3) << 8 |
            (unsigned char)(c4));
}

//parse the image files
inline idx3 parse_idx3(const std::vector<char>& raw){
    std::cout<<"Parsing the file as idx3 ..."<<std::flush;
    idx3 ans;
    ans.size=chars_to_int(raw[4], raw[5], raw[6], raw[7]);
    ans.rows=chars_to_int(raw[8], raw[9], raw[10], raw[11]);
    ans.cols=chars_to_int(raw[12], raw[13], raw[14], raw[15]);
    for (int i=16; i<raw.size(); i+=0){
        std::vector<char> to_add;
        for (int j=0; j<ans.rows; j++){
            for (int k=0; k<ans.cols; k++){
                to_add.push_back(raw[i]);
                i++;
            }
        }
        ans.images.push_back(to_add);
    }
    if (ans.images.size()!=ans.size){
        std::cout<<std::endl<<"\033[1;31mERROR: failed to parse file as idx3, corrupted file?\033[0m"<<std::endl;
        throw new std::exception();
    }
    std::cout<<" \033[1;32mDONE\033[0m"<<std::endl;
    std::cout<<"     "<<ans.size<<" images with resolution "<<ans.rows<<"x"<<ans.cols<<" pixels found"<<std::endl<<std::endl;
    return ans;
}

//parse the label files
inline idx1 parse_idx1(const std::vector<char>& raw){
    std::cout<<"Parsing the file as idx1 ..."<<std::flush;
    idx1 ans;
    ans.size=chars_to_int(raw[4], raw[5], raw[6], raw[7]);
    for (int i=8; i<raw.size(); i++){
        ans.labels.push_back(int((unsigned char)(raw[i])));
    }
    if (ans.labels.size()!=ans.size){
        std::cout<<std::endl<<"\033[1;31mERROR: failed to parse file as idx1, corrupted file?\033[0m"<<std::endl;
        throw new std::exception();
    }
    std::cout<<" \033[1;32mDONE\033[0m"<<std::endl;
    std::cout<<"     "<<ans.size<<" labels found"<<std::endl<<std::endl;
    return ans;
}

#endif //FILE_PARSER_H