#include "src/tensor.h"

using ::Tensor;

int main() {
    Tensor *t = new Tensor(3, 1, nullptr, new int[1]{3}, new double[3]{1, 2, 3});
}
