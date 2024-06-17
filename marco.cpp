#include <exception>
#include <string>
#include <iostream>

#define ASSIGN_STATUS_IF_CATCH_EXCEPTION(stmt, status_ref)                                           \
    do {                                                                                             \
        try {                                                                                        \
            { stmt; }                                                                                \
        } catch (std::exception& e) {                                                        \
            status_ref = "CATCH";   \
        }   \
    } while (0);                    \


#define ASSIGN_STATUS_IF_CATCH_EXCEPTION_NOT_REF(stmt, status_ref)                                           \
    do {                                                                                             \
        try {                                                                                        \
            { stmt; }                                                                                \
        } catch (std::exception& e) {                                                        \
            status_ref = "CATCH";   \
        }   \
    } while (0);                    \

void throwFunction(bool t) {
    if (t) {
        throw std::exception();
    }
}

int main() {
    std::string status("ORIGINAL");
    ASSIGN_STATUS_IF_CATCH_EXCEPTION(throwFunction(true), status);
    std::cout << status << std::endl;

    std::string status_not_ref("ORIGINAL");
    ASSIGN_STATUS_IF_CATCH_EXCEPTION_NOT_REF(throwFunction(true), status_not_ref);
    std::cout << status_not_ref << std::endl;
}