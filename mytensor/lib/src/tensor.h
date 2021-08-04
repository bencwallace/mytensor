#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
private:
    int *strides;
    int *shape;
    double* data;

    void generate_strides();
    int idx_to_pos(int *idx);

public:
    int ndims;
    const int size;

    Tensor(int size, int ndims, int *strides, int *shape, double *data);
    ~Tensor();
    
    Tensor &operator=(double val);
    double *operator[](int*);

    double *flatten();
};

#endif
