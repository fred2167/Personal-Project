#include "Borrow.h"

BorrowFactory::BorrowFactory() {
  Transaction::registerTransactionFactory("B", this);
}

Transaction *BorrowFactory::create() const { return new Borrow(); }

namespace {
// call constructor to register borrowFactory to the static factories in base
// class
BorrowFactory borrowFactory;
} // namespace
//======================================================================
istream &Borrow::read(istream &is) {
  // ID MediaType MovieType (movie sorting attributes)
  is >> customerID;
  is >> mediaType;
  is >> movieType;
  toBorrow = Inventory::createInventory(movieType);
  if (toBorrow != nullptr) {
    toBorrow->transactionRead(is);
    validFlag = true;
  }
  return is;
}

ostream &Borrow::printer(ostream &os) const {
  if (!validFlag) {
    return os;
  }
  os << "*Borrow* " << customerID << ", ";
  toBorrow->transactionPrinter(os);
  return os;
}

Borrow::~Borrow() {
  delete toBorrow;
  toBorrow = nullptr;
}

bool Borrow::process(Store &s) {
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
  auto c = dynamic_cast<Customer *>(p);

  // retrieve inventoty
  auto &inventorySet = Inventory::getInventorySet(movieType);
  auto iter = inventorySet.find(toBorrow);

  if (iter == inventorySet.end()) {
    cout << "**Not in Inventory** " << *this;
    validFlag = false;
    return false;
  }

  // retrieve mediaType
  Inventory *item = *iter;
  if (!item->hasMedia(mediaType)) {
    cout << "**Invalid MediaType** " << mediaType << " for " << *item;
    validFlag = false;
    return false;
  }

  MediaType *m = item->getMedia(mediaType);

  if (!m->hasStock()) {
    cout << "**Out of Stock** " << *item;
    validFlag = false;
    return false;
  }

  // subtract stock, add to item history, customer history and customer's loan
  // history
  m->borrowItem();
  item->addToInventoryHistory(this);
  c->addToCustomerHistory(this);
  c->addToOnLoan(item);
  return true;
};