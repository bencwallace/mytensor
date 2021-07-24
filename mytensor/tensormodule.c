#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "tensormodule.h"


int extract_shape(int ndims, PyObject *shape_seq, int *shape) {
    int size = 1;
    if (shape == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (int i = 0; i < ndims; i++) {
        PyObject *item = PySequence_GetItem(shape_seq, i);
        if (PyLong_Check(item)) {
            shape[i] = PyLong_AsLong(item);
            size *= shape[i];
        }
        else {
            PyErr_SetString(PyExc_ValueError, "Expected `shape` sequence to consist of integers.");
            return -1;
        }
    }
    return size;
}


static PyObject *Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyTensorObject *self;

    PyObject *shape_seq;
    if (!PyArg_ParseTuple(args, "O", &shape_seq))
        return NULL;
    if (!PySequence_Check(shape_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected `shape` to be a sequence.");
        return NULL;
    }

    self = (PyTensorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ndims = PySequence_Size(shape_seq);

        self->shape = (int *) malloc(self->ndims * sizeof(int));
        if (self->shape == NULL) {
            PyErr_NoMemory();
            return NULL;
        }

        self->size = extract_shape(self->ndims, shape_seq, self->shape);
        if (self->size < 0)
            return NULL;

        self->data = (double *) malloc(self->size * sizeof(double));
        if (self->data == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (int i = 0; i < self->size; i++)
            self->data[i] = 0;
    }

    self->strides = NULL;
    return (PyObject *) self;
}


static void Tensor_dealloc(PyTensorObject *self) {
    Py_XDECREF(self->strides);
    Py_XDECREF(self->shape);
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Tensor",
    .tp_doc = "My tensor type",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Tensor_new,
    .tp_dealloc = (destructor) Tensor_dealloc,
};


static PyModuleDef lib_tensor_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "lib_tensor",
    .m_doc = "My tensor module",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_lib_tensor(void) {
    PyObject *m;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;
    
    m = PyModule_Create(&lib_tensor_module);
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
