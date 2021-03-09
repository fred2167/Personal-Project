#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>

using namespace std;

namespace util {

// can consume the delimiter at the end too
static string readNextItem(istream &io, char delimiter = ',') {
  const char space = ' ';
  string str;
  getline(io, str, delimiter);
  if (str.empty()) {
    return str;
  } else {
    return str.substr(str.find_first_not_of(space),
                      str.find_last_not_of(space) + 1);
  }
}

static void eatGarbage(istream &io, char delimiter = ',') {
  string garbage;
  getline(io, garbage, delimiter);
}

template <class InputIterator, class T>
static InputIterator find(InputIterator first, InputIterator last,
                          const T &val) {
  while (first != last) {
    if (**first == *val)
      return first;
    ++first;
  }
  return last;
}

} // namespace util
#endif