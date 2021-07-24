#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "tensormodule.h"


int extract_shape(int ndims, PyObject *shape_seq, int *shape) {
    if (ndims == 0) {
        return 0;
    }
    int size = 1;
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


int *generate_strides(int n, int *shape) {
    int *reversed_strides = (int *) malloc(n * sizeof(int));
    if (reversed_strides == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    reversed_strides[0] = 1;
    for (int i = 1; i < n; i++) {
        reversed_strides[i] = reversed_strides[i - 1] * shape[n - i];
    }

    int *strides = (int *) malloc(n * sizeof(int));
    if (strides == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n; i++)
        strides[i] = reversed_strides[n - i - 1];

    return strides;
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
            self->data[i] = i;

        self->strides = generate_strides(self->ndims, self->shape);
        if (self->strides == NULL)
            return NULL;
    }
    return (PyObject *) self;
}


static void Tensor_dealloc(PyTensorObject *self) {
    Py_XDECREF(self->strides);
    Py_XDECREF(self->shape);
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *Tensor_subscript(PyTensorObject *self, PyObject *idx_seq) {
    if (!PySequence_Check(idx_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected multi-index to be a sequence.");
        return NULL;
    }

    int n = PySequence_Size(idx_seq);
    if (n != self->ndims) {
        int ndims_str_len = 1 + self->ndims / 10;   // length of ndims as a string
        char *format = "Expected multi-index to have length %d";
        char *message = (char *) malloc((strlen(format) - 2 + ndims_str_len));
        sprintf(message, format, self->ndims);
        PyErr_SetString(PyExc_ValueError, message);
        return NULL;
    }

    int *idx = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(idx_seq, i);
        if (PyLong_Check(item))
            idx[i] = PyLong_AsLong(item);
        else {
            PyErr_SetString(PyExc_ValueError, "Expected `shape` sequence to consist of integers.");
            return NULL;
        }
    }

    int pos = 0;
    for (int i = 0; i < self->ndims; i++)
        pos += idx[i] * self->strides[i];

    int value = self->data[pos];
    return PyFloat_FromDouble(value);   // todo: return different view, not new object
}


static PyObject *Tensor_get_strides(PyTensorObject *self, PyObject *args) {
    PyObject *strides_list = PyList_New(self->ndims);
    for (int i = 0; i < self->ndims; i++)
        PyList_SetItem(strides_list, i, PyLong_FromLong(self->strides[i]));
    return (PyObject *) strides_list;
}


static PyMethodDef Tensor_methods[] = {
    {"get_strides", (PyCFunction) Tensor_get_strides, METH_VARARGS, "Get the tensor strides"},
};


static PyMemberDef Tensor_members[] = {
    {
        .name = "size",
        .type = T_INT,
        .offset = offsetof(PyTensorObject, size),
        .flags = READONLY,
        .doc = "Tensor size",
    },
    {"ndims", T_INT, offsetof(PyTensorObject, ndims), READONLY, "Number of dimensions"},
};


static PyMappingMethods Tensor_mapping_methods = {
    .mp_subscript = (binaryfunc) Tensor_subscript,
};


static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Tensor",
    .tp_doc = "My tensor type",
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Tensor_new,
    .tp_dealloc = (destructor) Tensor_dealloc,
    .tp_members = Tensor_members,
    .tp_methods = Tensor_methods,
    .tp_as_mapping = &Tensor_mapping_methods,
};


static PyModuleDef lib_tensor_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mytensor.lib.tensor",
    .m_doc = "My tensor module",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_tensor(void) {
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
