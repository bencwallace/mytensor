template <typename T>
void Tensor<T>::generate_strides() {
    // scalar case
    if (ndims < 1) {
        strides = new int[1]{0};
        return;
    }

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


template <typename T>
int Tensor<T>::idx_to_pos(int *idx) const {
    int pos = 0;
    for (int i = 0; i < ndims; i++)
        pos += idx[i] * strides[i];
    return pos;
}


// base constructor
template <typename T>
Tensor<T>::Tensor(int ndims, int *strides, int *shape, T *data):
strides(strides), shape(shape), data(data), ndims(ndims) {
    // default shape is all zeros
    if (this->shape == nullptr) {
        this->shape = new int[ndims];
        for (int i = 0; i < ndims; i++)
            this->shape[i] = 0;
    }

    // product of shape entries should equal size (except in degenerate case)
    if (ndims == 0)
        size = 0;
    else
        size = prod(ndims, this->shape);

    // generate default strides for given shape
    if (this->strides == nullptr)
        generate_strides();

    // // default data is all zeros
    // if (data == nullptr)
    //     for (int i = 0; i < size; i++)
    //         data[i] = 0;
}


// default constructor
template <typename T>
Tensor<T>::Tensor(): Tensor(0, nullptr, nullptr, nullptr) {}


// copy constructor
template <typename T>
Tensor<T>::Tensor(const Tensor<T> &other):
size(other.size), ndims(other.ndims) {
    if (ndims > 0) {
        strides = new int[ndims];
        memcpy(strides, other.strides, ndims * sizeof(int));

        shape = new int[ndims];
        memcpy(shape, other.shape, ndims * sizeof(int));
    }

    if (size > 0) {
        data = new T[size];
        memcpy(data, other.data, size * sizeof(T));
    }
}


// destructor
template <typename T>
Tensor<T>::~Tensor() {
    delete strides;
    delete shape;
    delete data;
}


// operators
template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other) {
    Tensor result = other;
    for (int i = 0; i < result.size; i++)
        result.data[i] += data[i];
    return result;
}


// other methods
template <typename T>
T *Tensor<T>::flatten() {
    return data;
}
