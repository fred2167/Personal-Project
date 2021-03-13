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
  // copy not allowed
  Return(const Return &rhs) = delete;
  // move not allowed
  Return(Return &&rhs) = delete;

  // assignment not allowed
  Return &operator=(const Return &rhs) = delete;

  // move assignment not allowed
  Return &operator=(const Return &&rhs) = delete;

  Return() = default;

  ~Return() override;

  bool process(Store &s) override;
};

#endif