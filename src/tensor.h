#ifndef TENSOR_H
#define TENSOR_H

#include <string.h>

#include "utils.h"

template <typename T>
class Tensor {
private:
    int *strides;
    int *shape;
    T* data;
    int ndims;
    int size;

    void generate_strides() {
        // scalar case
        if (ndims < 1)
            strides = new int[1]{0};
            return;

        // vector case
        strides = new int[ndims];
        // generate strides in reverse order
        strides[0] = 1;
        for (int i = 1; i < ndims; i++) {
            int next = strides[i - 1] * shape[ndims - i];
            strides[i] = next;
        }
        // reverse strides
        for (int i = 0; i < ndims / 2; i++) {
            int temp = strides[i];
            strides[i] = strides[ndims - i - 1];
            strides[ndims - i - 1] = temp;
        }
    }

    int idx_to_pos(int *idx) const {
        int pos = 0;
        for (int i = 0; i < ndims; i++)
            pos += idx[i] * strides[i];
        return pos;
    }


public:
    // constructors and destructor
   
    // base constructor
    Tensor(int ndims, int *strides, int *shape, T *data):
    strides(strides), shape(shape), data(data), ndims(ndims) {
        // default shape is all zeros
        if (shape == nullptr) {
            shape = new int[ndims];
            for (int i = 0; i < ndims; i++)
                shape[i] = 0;
        }

        // product of shape entries should equal size (except in degenerate case)
        if (ndims == 0)
            size = 0;
        else
            size = prod(ndims, shape);

        // generate default strides for given shape
        if (strides == nullptr)
            generate_strides();

        // // default data is all zeros
        // if (data == nullptr)
        //     for (int i = 0; i < size; i++)
        //         data[i] = 0;
    }

    // default constructor
    Tensor(): Tensor(0, nullptr, nullptr, nullptr) {}

    // copy constructor
    Tensor(const Tensor<T> &other):
    size(other.size), ndims(other.ndims) {
        strides = new int[ndims];
        memcpy(strides, other.strides, ndims * sizeof(int));

        shape = new int[ndims];
        memcpy(shape, other.shape, ndims * sizeof(int));

        data = new double[size];
        memcpy(data, other.data, size * sizeof(T));
    }

    ~Tensor() {
        delete strides;
        delete shape;
        delete data;
    }

    // operators
    // Tensor operator+(const Tensor&);

    // other methods
    T *flatten() {
        return data;
    }
};

#endif
