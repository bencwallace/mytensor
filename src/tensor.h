#ifndef TENSOR_H
#define TENSOR_H

class Tensor {
private:
    int *strides;
    int *shape;
    double* data;
    int ndims;
    int size;

    void generate_strides();
    int idx_to_pos(int *idx) const;

public:
    // constructors and destructor
    Tensor(int ndims, int *strides, int *shape, double *data);
    Tensor();
    Tensor(const Tensor&);
    ~Tensor();

    // operators
    double *operator[](int*);
    Tensor operator+(const Tensor&);

    // other methods
    double *flatten();
};

#endif
