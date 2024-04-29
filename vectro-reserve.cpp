#include <cstddef>
#include <memory>

class Investment {};
class Stock : public Investment{};
class Bond : public Investment{};
class RealEstate : public Investment{};

void makeLogEntry(Investment*) {}

template<typename... Ts>
auto makeInvestment(Ts&&... params) {
    auto delInvmt = [](Investment* pInvestment) {
        makeLogEntry(pInvestment);
        delete pInvestment;
    };

    std::unique_ptr<Investment, decltype(delInvmt)> pInv(nullptr, delInvmt);
}