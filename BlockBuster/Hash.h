#ifndef HASH_H
#define HASH_H

#include <list>
#include <vector>

using namespace std;

template <class T> class Hash {

public:
  Hash();
  ~Hash();

  void insert(int key, T &value);

  T retrieve(int key);

  bool has(int key);

private:
  using KeyValuePair = pair<int, T>;

  vector<list<KeyValuePair> *> table;
  int numBuckets;

  int hashFn(int key);
};

#endif