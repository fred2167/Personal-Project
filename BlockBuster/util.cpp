#include "util.h"

// can consume the delimiter at the end too
string util::readNextItem(istream &io, char delimiter = ',') {
  const char space = ' ';
  string str;
  getline(io, str, delimiter);
  if (str.empty()) {
    return str;
  }
  return str.substr(str.find_first_not_of(space),
                    str.find_last_not_of(space) + 1);
}

void util::eatGarbage(istream &io, char delimiter = ',') {
  string garbage;
  getline(io, garbage, delimiter);
}