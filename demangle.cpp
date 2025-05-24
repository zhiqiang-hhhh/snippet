#include <iostream>
#include <cstdlib>
#include <cxxabi.h>

struct empty { };

template <typename T, int N>
  struct bar { };

int main()
{
  int     status;
  char   *realname;

  // typeid
  bar<empty,17>          u;
  const std::type_info  &ti = typeid(u);

  realname = abi::__cxa_demangle(ti.name(), NULL, NULL, &status);
  std::cout << ti.name() << "\t=> " << realname << "\t: " << status << '\n';
  std::free(realname);
}