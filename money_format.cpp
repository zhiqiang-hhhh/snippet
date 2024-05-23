#include <iostream>
#include <iomanip>
#include <locale>
#include <sstream>

// 自定义 facet 用于格式化货币输出
struct MoneyWithCommas : std::numpunct<char> {
    char do_thousands_sep() const override { return ','; }  // 定义千位分隔符为逗号
    std::string do_grouping() const override { return "\03"; }  // 定义千位分隔模式为每三位一个逗号
};

// 格式化浮点数为货币形式的字符串
std::string formatCurrency(double amount) {
    std::stringstream ss;
    ss.imbue(std::locale("")); // 使用默认本地化设置
    ss << amount; // 设置浮点数输出格式为固定小数位数（两位小数）

    return ss.str();
}

int main() {
    double amount = 1234;

    // 设置本地化，以便格式化货币
    std::locale loc(std::locale(), new MoneyWithCommas);

    // 获取本地化的货币输出器
    const std::money_put<char>& mp = std::use_facet<std::money_put<char>>(loc);

    // 格式化浮点数为货币形式的字符串
    std::string formattedAmount = formatCurrency(amount);

    // 输出格式化后的货币形式字符串
    std::cout.imbue(loc);
    mp.put(std::cout, false, std::cout, ' ', formattedAmount.c_str());

    std::cout << std::endl;

    return 0;
}
