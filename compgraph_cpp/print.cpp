#include <iostream>
#include <string>

using std::cout;
using std::endl;

void print(const char* x) {
  cout << x << endl;
}

void print(const std::string& s) {
  cout << s << endl;
}

void print(const void* x) {
  cout << x << endl;
}
