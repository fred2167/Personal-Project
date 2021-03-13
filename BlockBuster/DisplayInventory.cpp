#include "DisplayInventory.h"

DisplayInventoryFactory::DisplayInventoryFactory() {
  Transaction::registerTransactionFactory("I", this);
}

Transaction *DisplayInventoryFactory::create() const {
  return new DisplayInventory();
}

namespace {
// call constructor to register DisplayInventoryFactory to the static factories
// in base class
DisplayInventoryFactory displayInventoryFactory;
} // namespace
//======================================================================
istream &DisplayInventory::read(istream &is) {
  validFlag = true;
  return is;
}

ostream &DisplayInventory::printer(ostream &os) const {
  if (!validFlag) {
    return os;
  }
  os << "*DisplayInventory*" << endl;
  return os;
}

bool DisplayInventory::process(Store & /*s*/) {
  if (!validFlag) {
    return false;
  }
  cout << *this;
  Inventory::displayInventories();

  // return false so store process will delete transaction after process
  return false;
};