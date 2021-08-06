#ifndef TENSOR_H
#define TENSOR_H

#include <string.h>

#include "tensor_utils.h"

template <typename T>
class Tensor {
private:
    int *strides;
    int *shape;
    T* data;
    int ndims;
    int size;

    void generate_strides();
    int idx_to_pos(int *idx) const;

public:
    // static constructor
    template <typename S>
    static Tensor<S> *constant(S value, int ndims, int *shape) {
        int size = prod(ndims, shape);
        T *data = new S[size];
        for (int i = 0; i < size; i++)
            data[i] = value;
        return new Tensor(ndims, nullptr, shape, data);
    }

    // constructors and destructor
    Tensor(int ndims, int *strides, int *shape, T *data);
    Tensor();
    Tensor(const Tensor<T> &other);
    ~Tensor();

    // operators
    Tensor<T> operator+(const Tensor<T> &other);

    // other methods
    T *flatten();
};

#include "tensor.tpp"

#endif
