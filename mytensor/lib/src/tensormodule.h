#include <Python.h>

typedef struct
{
    PyObject_HEAD
    int size;
    int ndims;
    int *strides;
    int *shape;
    double* data;
    PyObject *base; // numpy's solution to views and reference counting
} PyTensorObject;
