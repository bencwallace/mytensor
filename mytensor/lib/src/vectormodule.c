#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "vectormodule.h"


// memory management
static void Vector_dealloc(PyVectorObject *self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyVectorObject *self;
    PyObject *data = NULL;
    int cuda = 0;

    // parse python object
    if (!PyArg_ParseTuple(args, "O|i", &data, &cuda)) {
        return NULL;
    }
    if (!PySequence_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence.");
        return NULL;
    }

    self = (PyVectorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->cuda = cuda;

        int size = PySequence_Size((PyObject *) data);
        if (!cuda) {
            self->data = (double *) malloc(size * sizeof(double));
        } else {
            // see https://stackoverflow.com/a/30249353/2449365
            PyErr_SetString(PyExc_NotImplementedError, "CUDA not implemented");
            return NULL;
        }
        if (self->data == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (int i = 0;  i < size; ++i) {
            PyObject *item = PySequence_GetItem(data, i);
            if (PyFloat_Check(item))
                self->data[i] = PyFloat_AsDouble(item);
            else if (PyLong_Check(item))
                self->data[i] = PyLong_AsDouble(item);
            else {
                Py_DECREF(item);
                PyErr_SetString(PyExc_ValueError, "Expected sequence elements to be ints or floats.");
                return NULL;
            }
            Py_DECREF(item);
            ++self->size;
        }
    }
    return (PyObject *) self;
}


// other methods
static PyObject *Vector_getsize(PyVectorObject *self, void *closure) {
    return PyLong_FromLong(self->size);
}

static PyObject *Vector_tolist(PyVectorObject *self) {
    PyObject *list = PyList_New(self->size);
    double *d = self->data;
    for (int i = 0; i < self->size; ++i) {
        PyList_SetItem(list, i, PyFloat_FromDouble(d[i]));
    }
    return (PyObject *) list;
}


static PyMethodDef Vector_methods[] = {
    {"size", (PyCFunction) Vector_getsize, METH_NOARGS, "Returns the size of the vector"},
    {"to_list", (PyCFunction) Vector_tolist, METH_NOARGS, "Converts the vector to a Python list"},
};


// sequence methods
static Py_ssize_t Vector_len(PyVectorObject *self) {
    return self->size;
}


static PySequenceMethods Vector_sequence_methods = {
    .sq_length = (lenfunc) Vector_len,
};


void add(int n, double *x, double *y, double *result) {
    for (int i = 0; i < n; i++)
        result[i] = x[i] + y[i];
}


// number methods
static PyObject *Vector_add(PyVectorObject *self, PyVectorObject *other) {
    int size = self->size;
    if (size != other->size) {
        printf("Size mismatch.");
        return NULL;
    }

    double sum[size];
    if (!self->cuda) {
        add(size, self->data, other->data, sum);
    } else {
        // see https://developer.nvidia.com/blog/even-easier-introduction-cuda/
        PyErr_SetString(PyExc_NotImplementedError, "CUDA implementation not available");
        return NULL;
    }
    PyObject *data_list = PyList_New(size);
    for (int i = 0; i < size; ++i) {
        PyList_SetItem(data_list, i, PyFloat_FromDouble(sum[i]));
    }

    PyObject *self_type = PyObject_Type((PyObject *) self);
    PyObject *args = Py_BuildValue("(O)", data_list);
    // todo: decrement data_list refcount?
    return PyObject_CallObject(self_type, args);
};


static PyNumberMethods Vector_number_methods = {
    .nb_add = (binaryfunc) Vector_add,
};


static PyMemberDef Vector_members[] = {
    {
        .name = "cuda",
        .type = T_BOOL,
        .offset = offsetof(PyVectorObject, cuda),
        .flags = READONLY,
        .doc = "Whether computations should be performed using CUDA",
    },
};



// Python type definition
static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Vector",
    .tp_doc = "My vector type",
    .tp_basicsize = sizeof(PyVectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Vector_new,
    .tp_dealloc = (destructor) Vector_dealloc,
    .tp_members = Vector_members,
    .tp_methods = Vector_methods,
    .tp_as_sequence = &Vector_sequence_methods,
    .tp_as_number = &Vector_number_methods,
};


// Python module definition and initialization
static PyModuleDef lib_vector_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mytensor.lib.vector",
    .m_doc = "My vector module",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_vector(void) {
    PyObject *m;
    if (PyType_Ready(&VectorType) < 0)
        return NULL;
    
    m = PyModule_Create(&lib_vector_module);
    if (m == NULL)
        return NULL;
    
    Py_INCREF(&VectorType);
    if (PyModule_AddObject(m, "Vector", (PyObject *) &VectorType) < 0) {
        Py_DECREF(&VectorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
