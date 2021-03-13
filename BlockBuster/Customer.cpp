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

void Customer::displayCustomerHistory() {
  // display from reversed ordered
  for (auto iter = history.rbegin(); iter != history.rend(); iter++) {
    cout << **iter;
  }
}

bool Customer::hasBorrow(Inventory *inv) {
  for (auto i : onLoan) {
    if (i->getType() == inv->getType() && *i == *inv) {
      return true;
    }
  }
  return false;
}

void Customer::addToCustomerHistory(Transaction *t) { history.push_back(t); }

void Customer::addToOnLoan(Inventory *inv) { onLoan.push_back(inv); }

void Customer::removeFromOnLoan(Inventory *inv) {
  for (auto iter = onLoan.begin(); iter != onLoan.end(); iter++) {
    auto onLoanPtr = *iter;
    if (onLoanPtr->getType() == inv->getType() && *onLoanPtr == *inv) {
      onLoan.erase(iter);
      return;
    }
  }
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
  return os;
}