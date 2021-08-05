template<typename T>
T prod(int n, T* values) {
    int result = 1;
    for (int i = 0; i < n; i++)
        result *= values[i];
    return result;
}
