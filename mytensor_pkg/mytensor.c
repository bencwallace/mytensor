#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "mytensor.h"


static void Vector_dealloc(PyVectorObject *self) {
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyVectorObject *self;
    PyListObject *data = NULL;

    // parse python object
    if (!PyArg_ParseTuple(args, "O", &data)) {
        // todo: proper error handling here and below
        printf("Failed to parse Python object.");
        return NULL;
    }

    if (!PyList_Check(data)) {
        printf("Constructor expected list argument.");
        return NULL;
    }

    self = (PyVectorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        int size = PyList_Size((PyObject *) data);
        self->data = (double *) malloc(size * sizeof(double));
        if (self->data == NULL) {
            printf("Failed to allocate vector data.");
            return NULL;
        }
        for (int i = 0;  i < size; ++i) {
            PyObject *item = PyList_GetItem((PyObject *) data, i);
            if (PyFloat_Check(item))
                self->data[i] = PyFloat_AsDouble(item);
            else if (PyLong_Check(item))
                self->data[i] = PyLong_AsDouble(item);
            else
                printf("Constructor expected list elements to be ints or floats.");
            ++self->size;
        }
        if (self->size != size) {
            printf("Something went wrong.");
            return NULL;
        }
    }
    return (PyObject *) self;
}


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


static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Vector",
    .tp_doc = "My vector type",
    .tp_basicsize = sizeof(PyVectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Vector_new,
    .tp_dealloc = (destructor) Vector_dealloc,
    .tp_methods = Vector_methods,
};

static PyModuleDef mytensormodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "mytensor",
    .m_doc = "My tensor module",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_mytensor(void) {
    PyObject *m;
    if (PyType_Ready(&VectorType) < 0)
        return NULL;
    
    m = PyModule_Create(&mytensormodule);
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
