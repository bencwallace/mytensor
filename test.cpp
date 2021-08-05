#include "src/tensor.h"

using ::Tensor;


int main() {
    Tensor t1;       // test default constructor
    Tensor *t2 = new Tensor(1, nullptr, nullptr, nullptr);   // test base constructor
    delete t2;      // test destructor

    return 0;
}
