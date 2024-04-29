#include <cstdlib>
#include <iostream>
#include <ctime>
 
int main() 
{
    std::srand(std::time(nullptr));
    int random_variable = std::rand(); 
 

    for (auto i = 0; i < 10; ++i) {
        auto r = std::rand();

        int a = 10;
        if (r != 0) {
            a = 0;
        }

        int b = 20;
        b = (r == 0) ? b : 0;
    }
}