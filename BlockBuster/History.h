#include "Customer.h"

#ifndef HISTORY_H
#define HISTORY_H

#include "Transaction.h"

using namespace std;

class HistoryFactory : public TransactionFactory {
public:
  HistoryFactory();

  Transaction *create() const override;
};

class History : public Transaction {

protected:
  unsigned int customerID = 0;

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;

public:
  // copy not allowed
  History(const History &rhs) = delete;

  // move not allowed
  History(History &&rhs) = delete;

  // assignment not allowed
  History &operator=(const History &rhs) = delete;

  // move assignment not allowed
  History &operator=(const History &&rhs) = delete;

  History() = default;

  ~History() override = default;

  bool process(Store &s) override;
};

#endif