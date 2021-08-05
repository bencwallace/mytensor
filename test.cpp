#include "src/tensor.h"

using ::Tensor;


int main() {
    Tensor<int> t1;       // test default constructor
    Tensor<double> *t2 = new Tensor<double>(1, nullptr, nullptr, nullptr);   // test base constructor
    delete t2;      // test destructor
    Tensor<int> *t3 = Tensor<int>::constant(42, 3, new int[3]{1, 2, 3});

    return 0;
}
