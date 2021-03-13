#include "Hash.h"

#ifndef STORE_H
#define STORE_H

#include "Inventory.h"
#include "MediaType.h"
#include "Person.h"
#include "Transaction.h"
#include "util.h"
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <string>

using namespace std;

class Transaction;

class Store {

public:
  // copy not allowed
  Store(const Store &rhs) = delete;

  // move not allowed
  Store(Store &&rhs) = delete;

  // assignment not allowed
  Store &operator=(const Store &rhs) = delete;

  // move assignment not allowed
  Store &operator=(const Store &&rhs) = delete;

  Store() = default;

  ~Store();

  void readInventory(const string &fileName, const string &mediaType);

  void readCustomer(const string &fileName);

  queue<Transaction *> readTransaction(const string &fileName);

  void processTransaction(queue<Transaction *> &q);

  bool hasCustomer(int id);

  Person *getCustomer(int id);

  void insertCustomer(int id, Person *p);

private:
  // map<int, Person *> customers;
  Hash customers;
};

#endif