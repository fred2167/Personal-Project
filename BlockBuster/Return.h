#include "Customer.h"

#ifndef RETURN_H
#define RETURN_H

#include "Transaction.h"

using namespace std;

class ReturnFactory : public TransactionFactory {
public:
  ReturnFactory();

  Transaction *create() const override;
};

class Return : public Transaction {

protected:
  unsigned int customerID = 0;
  string movieType = "";
  string mediaType = "";
  Inventory *toReturn = nullptr;

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;

public:
  ~Return() override;

  bool process(Store &s) override;
};

#endif