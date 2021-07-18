#include <Python.h>


typedef struct
{
    PyObject_HEAD
    int size;
    double* data;
} PyVectorObject;


static void Vector_dealloc(PyVectorObject *);
static PyObject *Vector_new(PyTypeObject *, PyObject *, PyObject *);
static PyObject *Vector_getsize(PyVectorObject *, void *);
static PyObject *Vector_tolist(PyVectorObject *);
PyMODINIT_FUNC PyInit_mytensor(void);
