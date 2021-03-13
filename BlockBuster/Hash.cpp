#include "Hash.h"

Hash::Hash() {
  numBuckets = 10;
  table.reserve(10);
  for (int i = 0; i < numBuckets; i++) {
    auto *l = new list<KeyValuePair>();
    table[i] = l;
  }
}
Hash::~Hash() {
  for (int i = 0; i < numBuckets; i++) {
    for (auto &pair : *table[i]) {
      delete pair.second;
    }
    delete table[i];
  }
}

void Hash::insert(int key, Person *&value) {
  if (has(key)) {
    return;
  }
  int bucket = hashFn(key);
  table[bucket]->push_front(make_pair(key, value));
}

Person *Hash::retrieve(int key) {
  int bucket = hashFn(key);
  for (auto &pair : *table[bucket]) {
    if (pair.first == key) {
      return pair.second;
    }
  }
  return nullptr;
}

bool Hash::has(int key) {
  int bucket = hashFn(key);
  for (KeyValuePair &pair : *table[bucket]) {
    if (pair.first == key) {
      return true;
    }
  }
  return false;
}

int Hash::hashFn(int key) { return key % numBuckets; }
