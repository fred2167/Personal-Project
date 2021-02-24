#include "autocomplete.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>

using namespace std;

void testBSTAll();

void Autocomplete::readFile(const string &fileName) {
  ifstream ifs(fileName);

  if (ifs.is_open()) {

    regex reg("[^ \t\\d]+.*|[0-9]+");
    string substr;

    BSTMap::mapped_type value;
    BSTMap::key_type key;

    // aborb number of entries at first line of the input file
    getline(ifs, substr);
    size_t entries = stoull(substr);
    while (getline(ifs, substr)) {

      auto iter = sregex_iterator(substr.begin(), substr.end(), reg);
      auto end = sregex_iterator();

      if (iter != end) {

        // cout << (*iter).str() << (*++iter).str() << endl;

        value = stoull((*iter).str());
        key = (*++iter).str();

        // cout << key << " " << value << endl;
        phrases[key] = value;
      }
    }

    ifs.close();
    phrases.rebalance();
    assert(phrases.size() == entries);
  } else {
    cout << "File cannot open!!" << endl;
  }
  // cout << phrases << endl;
}

bool Autocomplete::sortByWeight(BSTMap::value_type &a, BSTMap::value_type &b) {
  return a.second > b.second;
}

vector<BSTMap::value_type>
Autocomplete::complete(const BSTMap::key_type &prefix) const {
  auto v = phrases.getAll(prefix);
  sort(v.begin(), v.end(), sortByWeight);
  return v;
}
