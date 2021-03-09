#ifndef MYTEST_H
#define MYTEST_H

#include "Borrow.h"
#include "Comedy.h"
#include "Customer.h"
#include "Inventory.h"
#include "Movie.h"
#include "Person.h"
#include "Store.h"
#include "util.h"
#include <fstream>
#include <iostream>
#include <queue>
#include <vector>

using namespace std;

void testDVDInventory() {
  ifstream ifs("test.txt");
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return;
  }
  while (!ifs.eof()) {
    string op = util::readNextItem(ifs, ',');
    Inventory *inv = Inventory::createInventory(op);

    if (inv != nullptr) {
      MediaType *m = MediaType::createMediaType("D");
      ifs >> *m;
      inv->registerMediaType("D", m);
      util::eatGarbage(ifs, ',');
      ifs >> *inv;
      cout << *inv;
      delete inv;
    } else {
      string discard;
      getline(ifs, discard);
    }
  }
}
void testTransaction() {
  ifstream ifs("data4commands.txt");
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return;
  }
  while (!ifs.eof()) {
    string op;
    ifs >> op;
    Transaction *t = Transaction::createTransaction(op);
    if (t != nullptr) {

      ifs >> *t;
      cout << *t;
      delete t;
    } else {
      string discard;
      getline(ifs, discard);
    }
  }
}

void testCustomer() {
  ifstream ifs("data4customers.txt");
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return;
  }
  while (!ifs.eof()) {
    Person *c = Person::createPerson("C");
    ifs >> *c;

    cout << *c;
    delete c;
  }
}

void testStore() {
  Store s;
  s.readInventory("data4movies.txt", "D");
  s.readCustomer("data4customers.txt");
  queue<Transaction *> q = s.readTransaction("data4commands.txt");

  s.processTransaction(q);
}

void myTestAll() {
  // testDVDInventory();
  // testTransaction();
  // testCustomer();
  testStore();
}

#endif