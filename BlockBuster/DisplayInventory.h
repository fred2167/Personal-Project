#ifndef DISPLAYINVENTORY_H
#define DISPLAYINVENTORY_H

#include "Transaction.h"

using namespace std;

class DisplayInventoryFactory : public TransactionFactory {
public:
  DisplayInventoryFactory();

  Transaction *create() const override;
};

class DisplayInventory : public Transaction {

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;

public:
  // copy not allowed
  DisplayInventory(const DisplayInventory &rhs) = delete;

  // move not allowed
  DisplayInventory(DisplayInventory &&rhs) = delete;

  // assignment not allowed
  DisplayInventory &operator=(const DisplayInventory &rhs) = delete;

  // move assignment not allowed
  DisplayInventory &operator=(const DisplayInventory &&rhs) = delete;

  DisplayInventory() = default;

  ~DisplayInventory() override = default;

  bool process(Store &s) override;
};

#endif