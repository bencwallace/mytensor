#include "src/tensor.h"

using ::Tensor;


int main() {
    Tensor<int> t1;       // test default constructor
    auto *t2 = new Tensor<double>(
        1, nullptr, nullptr, nullptr
    );   // test base constructor
    delete t2;      // test destructor
    auto *t3 = Tensor<int>::constant(
        42, 3, new int[3]{1, 2, 3}
    );  // test static constructor
    auto t4 = *t3;   // test copy constructor
    auto t5 = t4 + t4;  // test add

    return 0;
}
