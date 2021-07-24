#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "tensormodule.h"


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
    free(reversed_strides);

    return strides;
}


int *extract_ints(int n, PyObject *py_int_seq) {
    int *int_seq = (int *) malloc(n * sizeof(int));
    if (int_seq == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(py_int_seq, i);
        if (!PyLong_Check(item)) {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError, "Expected sequence to contain integers.");
        }
        int_seq[i] = PyLong_AsLong(item);
        Py_DECREF(item);
    }
    return int_seq;
}


// todo: combine with extract_ints
double *extract_doubles(int n, PyObject *py_double_seq) {
    double *double_seq = (double *) malloc(n * sizeof(double));
    if (double_seq == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(py_double_seq, i);
        if (PyFloat_Check(item))
            double_seq[i] = PyFloat_AsDouble(item);
        else if (PyLong_Check(item))
            double_seq[i] = PyLong_AsDouble(item);
        else {
            Py_DECREF(item);
            PyErr_SetString(PyExc_TypeError, "Expected sequence to contain floats or ints.");
        }
        Py_DECREF(item);
    }
    return double_seq;
}


int prod(int n, int *seq) {
    if (n == 0) return 0;
    int result = 1;
    for (int i = 0; i < n; i++)
        result *= seq[i];
    return result;
}


static PyObject *Tensor_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyTensorObject *self;

    PyObject *shape_seq = NULL;
    PyObject *data_seq = NULL;
    if (!PyArg_ParseTuple(args, "O|O", &shape_seq, &data_seq))
        return NULL;

    if (!PySequence_Check(shape_seq)) {
        PyErr_SetString(PyExc_TypeError, "Expected `shape` to be a sequence.");
        return NULL;
    }
    int ndims = PySequence_Size(shape_seq);
    int *shape = extract_ints(ndims, shape_seq);
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
        data = extract_doubles(size, data_seq);
        if (data == NULL)
            return NULL;
    } else {
        data = (double *) malloc(size * sizeof(double));
        for (int i = 0; i < size; i++)
            data[i] = i;
    }

    self = (PyTensorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ndims = ndims;
        self->shape = shape;
        self->size = size;
        self->data = data;

        self->strides = generate_strides(self->ndims, self->shape);
        if (self->strides == NULL)
            return NULL;
    }
    return (PyObject *) self;
}


static void Tensor_dealloc(PyTensorObject *self) {
    free(self->strides);
    free(self->shape);
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


// todo: simplify
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
        free(message);
        PyErr_SetString(PyExc_ValueError, message);
        return NULL;
    }

    int *idx = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(idx_seq, i);
        if (PyLong_Check(item)) {
            idx[i] = PyLong_AsLong(item);
            Py_DECREF(item);
        }
        else {
            Py_DECREF(item);
            PyErr_SetString(PyExc_ValueError, "Expected `shape` sequence to consist of integers.");
            return NULL;
        }
    }

    int pos = 0;
    for (int i = 0; i < self->ndims; i++)
        pos += idx[i] * self->strides[i];
    free(idx);

    int value = self->data[pos];
    PyObject *shape_list = PyList_New(1);
    PyList_SetItem(shape_list, 0, PyLong_FromLong(1));

    PyObject *data_list = PyList_New(1);
    PyList_SetItem(data_list, 0, PyLong_FromLong(value));

    PyObject *args = Py_BuildValue("OO", shape_list, data_list);
    // todo: defined setitem and check if this is an actual view
    return PyObject_CallObject(PyObject_Type((PyObject *) self), args);
}


static PyObject *Tensor_get_strides(PyTensorObject *self, PyObject *args) {
    // todo: is reference counting needed?
    PyObject *strides_list = PyList_New(self->ndims);
    for (int i = 0; i < self->ndims; i++)
        PyList_SetItem(strides_list, i, PyLong_FromLong(self->strides[i]));
    return (PyObject *) strides_list;
}


static PyObject *Tensor_repr(PyTensorObject *self) {
    if (self->size > 1)
        return PyBaseObject_Type.tp_repr((PyObject *) self);
    return PyFloat_Type.tp_repr(PyFloat_FromDouble(self->data[0]));
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
    .tp_repr = (reprfunc) Tensor_repr,
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
