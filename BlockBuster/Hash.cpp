#include "Hash.h"

template <class T> Hash<T>::Hash() {
  numBuckets = 10;
  table.reserve(10);
  for (int i = 0; i < numBuckets; i++) {
    list<KeyValuePair> *l = new list<KeyValuePair>();
    table[i] = l;
  }
}
template <class T> Hash<T>::~Hash() {
  for (int i = 0; i < numBuckets; i++) {
    for (auto iter = table[i]->begin(); iter != table[i]->end(); iter++) {
      delete iter->second;
    }
    delete table[i];
  }
}

template <class T> void Hash<T>::insert(int key, T &value) {
  if (has(key)) {
    return;
  }
  int bucket = hashFn(key);
  table[bucket]->push_front(make_pair(key, value));
}

template <class T> T Hash<T>::retrieve(int key) {
  int bucket = hashFn(key);
  for (auto &pair : *table[bucket]) {
    if (pair.first == key) {
      return pair.second;
    }
  }
  T dummy;
  return dummy;
}

template <class T> bool Hash<T>::has(int key) {
  int bucket = hashFn(key);
  for (KeyValuePair &pair : *table[bucket]) {
    if (pair.first == key) {
      return true;
    }
  }
  return false;
}

template <class T> int Hash<T>::hashFn(int key) { return key % numBuckets; }
