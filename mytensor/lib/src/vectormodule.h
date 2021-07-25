#include <stdbool.h>
#include <Python.h>


typedef struct
{
    PyObject_HEAD
    bool cuda;
    int size;
    double* data;
} PyVectorObject;

// memory management
static void Vector_dealloc(PyVectorObject *);
static PyObject *Vector_new(PyTypeObject *, PyObject *, PyObject *);

// sequence methods
static Py_ssize_t Vector_len(PyVectorObject *);

// number methods
static PyObject *Vector_add(PyVectorObject *, PyVectorObject *);

// other methods
static PyObject *Vector_getsize(PyVectorObject *, void *);
static PyObject *Vector_tolist(PyVectorObject *);

// module initializer
PyMODINIT_FUNC PyInit_mytensor(void);
