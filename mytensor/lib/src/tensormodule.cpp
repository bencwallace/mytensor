#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include <iostream>
#include <sstream>

#include "tensormodule.h"


template <typename T>
T extract_item(PyObject *item) {}

template <>
double extract_item(PyObject *item) {
    if (PyFloat_Check(item))
        return PyFloat_AsDouble(item);
    else if PyLong_Check(item)
        return PyLong_AsDouble(item);
    PyErr_SetString(PyExc_TypeError, "Expected sequence to contain floats or ints.");
    Py_DECREF(item);
    return 0;
}

template <>
int extract_item(PyObject *item) {
    if (PyLong_Check(item))
        return PyLong_AsLong(item);
    PyErr_SetString(PyExc_TypeError, "Expected sequence to contain ints.");
    Py_DECREF(item);
    return 0;
}


template <typename T>
T *extract_seq(int n, PyObject *py_seq) {
    T *seq = new T[n];
    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(py_seq, i);
        T value = extract_item<T>(item);
        if (PyErr_Occurred())
            return NULL;
        seq[i] = value;
    }
    return seq;
}


int prod(int n, int *seq) {
    if (n == 0) return 0;
    int result = 1;
    for (int i = 0; i < n; i++)
        result *= seq[i];
    return result;
}


PyObject *new_tensor(
    PyTypeObject *type,
    int ndims,
    int size,
    int *strides,
    int *shape,
    double *data,
    PyObject *base
) {
    PyTensorObject *self = (PyTensorObject *) type->tp_alloc(type, 0);
    if (base != NULL)
        Py_INCREF(base);
    self->tensor = new Tensor(size, ndims, strides, shape, data);
    self->base = base;
    return (PyObject *) self;
}


static PyObject *Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject *shape_seq = NULL;
    PyObject *data_seq = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &shape_seq, &data_seq))
        return NULL;

    if (!PySequence_Check(shape_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected `shape` to be a sequence.");
        return NULL;
    }
    int ndims = PySequence_Size(shape_seq);
    int *shape = extract_seq<int>(ndims, shape_seq);
    if (shape == NULL)
        return NULL;
    int size = prod(ndims, shape);

    double *data;
    if (data_seq != NULL) {
        if (!PySequence_Check(data_seq)) {
            PyErr_SetString(PyExc_TypeError, "Expected `data` to be a sequence.");
            return NULL;
        }
        if (PySequence_Length(data_seq) != size) {
            PyErr_SetString(PyExc_ValueError, "`shape` and `data` are incompatible.");
            return NULL;
        }
        data = extract_seq<double>(size, data_seq);
        if (data == NULL)
            return NULL;
    } else {
        data = new double[size];
        for (int i = 0; i < size; i++)
            data[i] = i;
    }

    return new_tensor(type, ndims, size, NULL, shape, data, NULL);
}


static void Tensor_dealloc(PyTensorObject *self) {
    if (self->base == NULL)
        delete self->tensor;
    else
        Py_DECREF(self->base);
    Py_TYPE(self)->tp_free((PyObject *) self);
}



static PyObject *Tensor_subscript(PyTensorObject *self, PyObject *idx_seq) {
    if (!PySequence_Check(idx_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected multi-index to be a sequence.");
        return NULL;
    }
    int n = PySequence_Size(idx_seq);
    if (n != self->tensor->ndims) {
        std::ostringstream message;
        message << "Expected multi-index to have length " << self->tensor->ndims;
        PyErr_SetString(PyExc_ValueError, message.str().c_str());
        return NULL;
    }
    int *idx = extract_seq<int>(n, idx_seq);
    if (idx == NULL)
        return NULL;
    double *data = (*self->tensor)[idx];
    delete idx;

    int *shape = new int[1]{1};
    PyTypeObject *type = Py_TYPE(self);
    PyObject *result = new_tensor(type, 1, 1, NULL, shape, data, (PyObject *) self);
    return result;
}


static int Tensor_ass_subscript(PyTensorObject *self, PyObject *idx_seq, PyObject *py_val) {
    if (!PySequence_Check(idx_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected multi-index to be a sequence.");
        return -1;
    }
    int n = PySequence_Size(idx_seq);
    if (n != self->tensor->ndims) {
        std::ostringstream message;
        message << "Expected multi-index to have length " << self->tensor->ndims;
        PyErr_SetString(PyExc_ValueError, message.str().c_str());
        return -1;
    }
    int *idx = extract_seq<int>(n, idx_seq);
    if (idx == NULL)
        return -1;

    if (!PyFloat_Check(py_val)) {
        PyErr_SetString(PyExc_ValueError, "Expected assigned value to be float.");
        return -1;
    }
    double value = PyFloat_AsDouble(py_val);
    *(*self->tensor)[idx] = value;
    return 0;
}


// static PyObject *Tensor_get_strides(PyTensorObject *self, PyObject *args) {
//     // todo: is reference counting needed?
//     PyObject *strides_list = PyList_New(self->ndims);
//     for (int i = 0; i < self->ndims; i++)
//         PyList_SetItem(strides_list, i, PyLong_FromLong(self->strides[i]));
//     return (PyObject *) strides_list;
// }


static PyObject *Tensor_get_size(PyTensorObject *self, PyObject *args) {
    return PyLong_FromLong(self->tensor->size);
}


static PyObject *Tensor_flatten(PyTensorObject *self, PyObject *args) {
    double *result = self->tensor->flatten();
    PyObject *py_result = PyList_New(self->tensor->size);
    for (int i = 0; i < self->tensor->size; i++)
        PyList_SetItem(py_result, i, PyFloat_FromDouble(result[i]));
    return py_result;
}


static PyMethodDef Tensor_methods[] = {
    {"size", (PyCFunction) Tensor_get_size, METH_VARARGS, "Get the tensor size."},
    {"flatten", (PyCFunction) Tensor_flatten, METH_VARARGS, "Flatten into a list."},
    // {"get_strides", (PyCFunction) Tensor_get_strides, METH_VARARGS, "Get the tensor strides"},
    {NULL, NULL, NULL, NULL},
};


static PyMappingMethods Tensor_mapping_methods = {
    NULL,   // length
    (binaryfunc) Tensor_subscript,  // subscript
    .mp_ass_subscript = (objobjargproc) Tensor_ass_subscript,
};


static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "mytensor.Tensor",                          // name
    sizeof(PyTensorObject),                     // basicsize
    0,                                          // itemsize
    (destructor) Tensor_dealloc,                // dealloc
};


static PyModuleDef tensor_module = {
    PyModuleDef_HEAD_INIT,
    "mytensor.lib.tensor",                      // name
    "My tensor module",                         // doc
    -1,                                         // size
    NULL, NULL, NULL, NULL,
};


PyMODINIT_FUNC PyInit_tensor(void) {
    PyObject *m;

    TensorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    TensorType.tp_doc = "My tensor type";
    TensorType.tp_new = Tensor_new;
    // TensorType.tp_str = (reprfunc) Tensor_str;
    TensorType.tp_as_mapping = &Tensor_mapping_methods;
    TensorType.tp_methods = Tensor_methods;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;
    
    m = PyModule_Create(&tensor_module);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *) &TensorType) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

