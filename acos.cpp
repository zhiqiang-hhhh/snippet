#include <iostream>
#include <cmath>  // 包含 acos 函数的头文件

int main() {
    double value = 0.5;  // 输入值
    double angle = std::acos(value);  // 计算反余弦值
    
    // 输出结果
    std::cout << "The arccosine of " << value << " is " << angle << " radians." << std::endl;


   value = 1.1;
    angle = std::acos(value);  // 计算反余弦值
    std::cout << "The arccosine of " << value << " is " << angle << " radians." << std::endl;
    
    return 0;
}
