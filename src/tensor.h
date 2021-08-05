#ifndef TENSOR_H
#define TENSOR_H

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
    // constructors and destructor
    Tensor(int ndims, int *strides, int *shape, T *data);
    Tensor();
    Tensor(const Tensor&);
    ~Tensor();

    // operators
    // Tensor operator+(const Tensor&);

    // other methods
    T *flatten();
};

#endif
