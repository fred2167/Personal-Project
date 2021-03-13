#include "Person.h"
#ifndef HASH_H
#define HASH_H

#include <list>
#include <vector>

using namespace std;

class Hash {

public:
  // copy not allowed
  Hash(const Hash &rhs) = delete;

  // move not allowed
  Hash(Hash &&rhs) = delete;

  // assignment not allowed
  Hash &operator=(const Hash &rhs) = delete;

  // move assignment not allowed
  Hash &operator=(const Hash &&rhs) = delete;

  Hash();

  ~Hash();

  void insert(int key, Person *&value);

  Person *retrieve(int key);

  bool has(int key);

private:
  using KeyValuePair = pair<int, Person *>;

  vector<list<KeyValuePair> *> table;
  int numBuckets;

  int hashFn(int key);
};

#endif