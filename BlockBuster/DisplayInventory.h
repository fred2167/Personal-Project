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
  ~DisplayInventory() = default;

  bool process(Store &s) override;
};

#endif