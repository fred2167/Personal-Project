#include "Customer.h"

CustomerFactory::CustomerFactory() {
  // identifer: C as customer
  Person::registerPersonFactory("C", this);
}

Person *CustomerFactory::create() { return new Customer(); }

namespace {
// call constructor to register customerFactory to the static factories in base
// class
CustomerFactory customerFactory;
} // namespace
//===========================================================

Customer::~Customer() {
  for (Transaction *t : history) {
    delete t;
    t = nullptr;
  }
}

void Customer::displayCutomerHistory() {
  // display from reversed ordered
  for (auto iter = history.rbegin(); iter != history.rend(); iter++) {
    cout << **iter;
  }
}

void Customer::addToCustomerHistory(Transaction *t) { history.push_back(t); }

void Customer::addToOnLoan(Inventory *inv) { onLoan.push_back(inv); }

void Customer::removeFromOnLoan(Inventory *inv) {
  auto iter = util::find(onLoan.begin(), onLoan.end(), inv);
  onLoan.erase(iter);
}

void Customer::addToAlreadyReturn(Inventory *inv) {
  alreadyReturn.push_back(inv);
}

istream &Customer::read(istream &is) {
  is >> id;
  name = util::readNextItem(is, '\n');
  return is;
}

ostream &Customer::printer(ostream &os) const {
  os << id << ", " << name << endl;
  for (Transaction *t : history) {
    os << *t << endl;
  }
  return os;
}