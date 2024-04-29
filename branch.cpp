#include <vector>


// SELECT oid FROM table WHERE col < X
int main() {
    const std::size_t N = 10000;
    const int V = 0;
    std::vector<int> src1;
    std::vector<int> src2;
    std::vector<int> res;
    for (std::size_t i = 0, j = 0; i < N; ++i) {
        /*branch version*/
        if (src1[i] < V)
            src2[j++] = i;
        /*predict version*/
        bool b = (src1[i] < V);
        src2[j] = i;
        j += b;
    }
}