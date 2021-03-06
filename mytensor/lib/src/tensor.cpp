#include <iostream>
#include <string.h>

#include "tensor.h"

using namespace std;


Tensor::Tensor(const Tensor &other):
size(other.size), ndims(other.ndims) {
    strides = new int[ndims];
    memcpy(strides, other.strides, ndims * sizeof(int));

    shape = new int[ndims];
    memcpy(shape, other.shape, ndims * sizeof(int));

    data = new double[size];
    memcpy(data, other.data, size * sizeof(double));
}

Tensor::Tensor(int size, int ndims, int *strides, int *shape, double *data):
strides(strides), shape(shape), data(data), ndims(ndims), size(size) {
    if (strides == nullptr)
        generate_strides();
};

Tensor::~Tensor() {
    delete strides;
    delete shape;
    delete data;
}

void Tensor::generate_strides() {
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

int Tensor::idx_to_pos(int *idx) const {
    int pos = 0;
    for (int i = 0; i < ndims; i++)
        pos += idx[i] * strides[i];
    return pos;
}

double *Tensor::operator[](int *idx) {
    return data + idx_to_pos(idx);
}

Tensor Tensor::operator+(const Tensor &other) {
    Tensor result = other;
    for (int i = 0; i < result.size; i++)
        result.data[i] += data[i];
    return result;
}

double *Tensor::flatten() {
    return data;
}
