#include "Customer.h"

#ifndef BORROW_H
#define BORROW_H

#include "Inventory.h"
#include "Transaction.h"

using namespace std;

class Borrow;

class BorrowFactory : public TransactionFactory {
public:
  BorrowFactory();

  Transaction *create() const override;
};

class Borrow : public Transaction {

protected:
  unsigned int customerID = 0;
  string movieType = "";
  string mediaType = "";
  Inventory *toBorrow = nullptr;

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;

public:
  // copy not allowed
  Borrow(const Borrow &rhs) = delete;

  // move not allowed
  Borrow(Borrow &&rhs) = delete;

  // assignment not allowed
  Borrow &operator=(const Borrow &rhs) = delete;

  // move assignment not allowed
  Borrow &operator=(const Borrow &&rhs) = delete;

  Borrow() = default;

  ~Borrow() override;

  bool process(Store &s) override;
};

#endif