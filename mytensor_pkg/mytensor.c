#define PY_SSIZE_T_CLEAN
#include <Python.h>


/* todo: consider putting these in a header file */
typedef struct
{
    PyObject_HEAD
} PyVectorObject;


static PyTypeObject VectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "mytensor.Vector",
    .tp_doc = "My vector type",
    .tp_basicsize = sizeof(PyVectorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
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
