#include "Store.h"

Store::~Store() {
  // for (auto &pair : customers) {
  //   delete pair.second;
  //   pair.second = nullptr;
  // }
  Inventory::deleteInventories();
}

void Store::readInventory(const string &fileName, const string &mediaType) {
  ifstream ifs(fileName);
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return;
  }
  while (!ifs.eof()) {
    string op = util::readNextItem(ifs, ',');
    Inventory *inv = Inventory::createInventory(op);

    if (inv != nullptr) {
      MediaType *m = MediaType::createMediaType(mediaType);
      ifs >> *m;
      inv->registerMediaType(mediaType, m);
      util::eatGarbage(ifs, ',');
      ifs >> *inv;
      // cout << *inv;
      Inventory::insertInventory(op, inv);
    } else {
      util::eatGarbage(ifs, '\n');
    }
  }
}

void Store::readCustomer(const string &fileName) {
  ifstream ifs(fileName);
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return;
  }
  while (!ifs.eof()) {
    Person *c = Person::createPerson("C");
    ifs >> *c;
    // cout << *c;
    insertCustomer(c->getId(), c);
  }
}

queue<Transaction *> Store::readTransaction(const string &fileName) {
  ifstream ifs(fileName);
  queue<Transaction *> q;
  if (!ifs.is_open()) {
    cerr << "File cannot open!" << endl;
    return q;
  }

  while (!ifs.eof()) {
    string op;
    ifs >> op;
    if (op.empty()) {
      util::eatGarbage(ifs, '\n');
      continue;
    }
    Transaction *t = Transaction::createTransaction(op);
    if (t != nullptr) {
      ifs >> *t;
      // cout <<*t;
      q.push(t);
    } else {
      delete t;
      util::eatGarbage(ifs, '\n');
    }
  }

  return q;
}

void Store::processTransaction(queue<Transaction *> &q) {

  while (!q.empty()) {
    Transaction *t = q.front();
    q.pop();
    bool flag = t->process(*this);
    if (!flag) {
      delete t;
    }
  }
}

bool Store::hasCustomer(int id) {
  // return customers.count(id) == 1;
  return customers.has(id);
}

Person *Store::getCustomer(int id) {
  // return customers.at(id);
  return customers.retrieve(id);
}

void Store::insertCustomer(int id, Person *p) {
  // customers.emplace(id, p);
  customers.insert(id, p);
}