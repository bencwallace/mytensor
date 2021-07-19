#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "mytensor.h"


// memory management
static void Vector_dealloc(PyVectorObject *self) {
    Py_XDECREF(self->data);
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyObject *Vector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyVectorObject *self;
    PyListObject *data = NULL;

    // parse python object
    if (!PyArg_ParseTuple(args, "O", &data)) {
        return NULL;
    }

    // todo: accept general sequences
    if (!PyList_Check(data)) {
        PyErr_SetString(PyExc_TypeError, "Expected a string.");
        return NULL;
    }

    self = (PyVectorObject *) type->tp_alloc(type, 0);
    if (self != NULL) {
        int size = PyList_Size((PyObject *) data);
        self->data = (double *) malloc(size * sizeof(double));
        if (self->data == NULL) {
            PyErr_NoMemory();
            return NULL;
        }
        for (int i = 0;  i < size; ++i) {
            PyObject *item = PyList_GetItem((PyObject *) data, i);
            if (PyFloat_Check(item))
                self->data[i] = PyFloat_AsDouble(item);
            else if (PyLong_Check(item))
                self->data[i] = PyLong_AsDouble(item);
            else {
                PyErr_SetString(PyExc_ValueError, "Expected list elements to be ints or floats.");
                return NULL;
            }
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


// number methods
static PyObject *Vector_add(PyVectorObject *self, PyVectorObject *other) {
    int size = self->size;
    if (size != other->size) {
        printf("Size mismatch.");
        return NULL;
    }

    PyObject *data_list = PyList_New(size);
    for (int i = 0; i < size; ++i) {
        double sum = self->data[i] + other->data[i];
        PyList_SetItem(data_list, i, PyFloat_FromDouble(sum));
    }

    PyObject *self_type = PyObject_Type((PyObject *) self);
    PyObject *args = Py_BuildValue("(O)", data_list);
    return PyObject_CallObject(self_type, args);
};


static PyNumberMethods Vector_number_methods = {
    .nb_add = (binaryfunc) Vector_add,
};


// Python type definition
PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Vector",
    .tp_doc = "My vector type",
    .tp_basicsize = sizeof(PyVectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Vector_new,
    .tp_dealloc = (destructor) Vector_dealloc,
    .tp_methods = Vector_methods,
    .tp_as_sequence = &Vector_sequence_methods,
    .tp_as_number = &Vector_number_methods,
};


// Python module definition and initialization
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
