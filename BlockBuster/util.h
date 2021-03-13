#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>

using namespace std;

namespace util {

// can consume the delimiter at the end too
string readNextItem(istream &io, char delimiter);

void eatGarbage(istream &io, char delimiter);

} // namespace util
#endif