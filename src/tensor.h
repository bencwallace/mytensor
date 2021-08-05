#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
private:
    int *strides;
    int *shape;
    double* data;

    void generate_strides();
    int idx_to_pos(int *idx) const;

public:
    int ndims;
    const int size;

    Tensor(const Tensor&);
    Tensor(int size, int ndims, int *strides, int *shape, double *data);
    ~Tensor();
    
    double *operator[](int*);
    Tensor operator+(const Tensor&);

    double *flatten();
};

#endif
