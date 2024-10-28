#include <iostream>

/// Implement method to obtain an address of 'add' function.
template <typename Derived, typename NullablePropertyArg>
class IAggregateFunctionHelper {
    public:
    using NullableProperty = NullablePropertyArg;

    virtual void print() {
        if constexpr (std::is_same_v<NullableProperty, bool>) {
            std::cout << "bool" << std::endl;
        } else {
            std::cout << "not bool" << std::endl;
        }
    }
};

class AggFunc1 : public IAggregateFunctionHelper<AggFunc1, bool> {
    // ...
};

template <typename NestedFunction>
class Helper2 : public IAggregateFunctionHelper<NestedFunction, typename NestedFunction::NullableProperty> {
    // ...
};


int main() {
    auto tmp = Helper2<AggFunc1>();
    tmp.print();
    return 0;
}