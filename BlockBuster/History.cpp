#include "History.h"

HistoryFactory::HistoryFactory() {
  Transaction::registerTransactionFactory("H", this);
}

Transaction *HistoryFactory::create() const { return new History(); }

namespace {
// call constructor to register historyFactory to the static factories in base
// class
HistoryFactory historyFactory;
} // namespace
//======================================================================
istream &History::read(istream &is) {
  // ID
  is >> customerID;
  validFlag = true;
  return is;
}

ostream &History::printer(ostream &os) const {
  if (!validFlag) {
    return os;
  }
  os << "*History* Customer: " << customerID << endl;
  return os;
}

bool History::process(Store &s) {
  if (!validFlag) {
    return false;
  }

  // retrieve customer
  if (!s.hasCustomer(customerID)) {
    cout << "**Invalid CustomerID** " << customerID << endl;
    validFlag = false;
    return false;
  }
  Person *p = s.getCustomer(customerID);
  Customer *c = dynamic_cast<Customer *>(p);

  cout << *this;
  c->displayCutomerHistory();

  // return false so store process will delete transaction after process
  return false;
};