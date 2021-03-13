#include "Return.h"

ReturnFactory::ReturnFactory() {
  Transaction::registerTransactionFactory("R", this);
}

Transaction *ReturnFactory::create() const { return new Return(); }

namespace {
// call constructor to register returnFactory to the static factories in base
// class
ReturnFactory returnFactory;
} // namespace

//======================================================================
istream &Return::read(istream &is) {
  // ID MediaType MovieType (movie sorting attributes)
  is >> customerID;
  is >> mediaType;
  is >> movieType;
  toReturn = Inventory::createInventory(movieType);
  if (toReturn != nullptr) {
    toReturn->transactionRead(is);
    validFlag = true;
  }
  return is;
}

ostream &Return::printer(ostream &os) const {
  if (!validFlag) {
    return os;
  }
  os << "*Return* " << customerID << ", ";
  toReturn->transactionPrinter(os);
  return os;
}

Return::~Return() {
  delete toReturn;
  toReturn = nullptr;
}

bool Return::process(Store &s) {
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
  auto *c = dynamic_cast<Customer *>(p);

  // Check if customer has borrow the item
  if (!c->hasBorrow(toReturn)) {
    cout << "**Customer " << customerID << " NOT borrow** ";
    toReturn->transactionPrinter(cout);
    validFlag = false;
    return false;
  }

  // retrieve inventoty
  auto &inventorySet = Inventory::getInventorySet(movieType);
  auto iter = inventorySet.find(toReturn);

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

  // add stock
  m->returnItem();

  // keep track histories
  item->addToInventoryHistory(this);
  c->addToCustomerHistory(this);
  c->addToAlreadyReturn(item);
  c->removeFromOnLoan(item);
  //   cout <<*this;
  return true;
};