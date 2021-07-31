#ifndef TENSORMODULE_H
#define TENSORMODULE_H

#include <Python.h>

#include "tensor.h"

typedef struct
{
    PyObject_HEAD
    Tensor *tensor;
    PyObject *base;
} PyTensorObject;

#endif
