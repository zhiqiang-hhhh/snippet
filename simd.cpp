#include <immintrin.h>
#include <stdio.h>

int main() {
    double data[2] = {2.5, 2.7};
    __m128d a = _mm_load_pd(data);
    __m128d result = _mm_round_pd(a, _MM_FROUND_TO_POS_INF);

    double rounded[2];
    _mm_store_pd(rounded, result);

    printf("Rounded values: %lf, %lf\n", rounded[0], rounded[1]);

    return 0;
}

