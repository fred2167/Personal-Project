/**
 * Driver for starting movie store tests
 */

#include <iostream>

using namespace std;

// forward declaration, implementation in store_test.cpp
void testAll();
void myTestAll();

int main() {
  // testAll();
  myTestAll();
  cout << "Done." << endl;
  return 0;
}
