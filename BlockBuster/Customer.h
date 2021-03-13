#ifndef CUSTOMER_H
#define CUSTOMER_H

#include "Borrow.h"
#include "Inventory.h"
#include "Person.h"
#include "Transaction.h"
#include "util.h"
#include <vector>

using namespace std;

class Inventory;

class CustomerFactory : public PersonFactory {

public:
  CustomerFactory();
  Person *create() override;
};

class Customer : public Person {

protected:
  vector<Transaction *> history;
  vector<Inventory *> onLoan;
  vector<Inventory *> alreadyReturn;

public:
  // copy not allowed
  Customer(const Customer &rhs) = delete;

  // move not allowed
  Customer(Customer &&rhs) = delete;

  // assignment not allowed
  Customer &operator=(const Customer &rhs) = delete;

  // move assignment not allowed
  Customer &operator=(const Customer &&rhs) = delete;

  Customer() = default;

  ~Customer() override;

  void displayCustomerHistory();

  bool hasBorrow(Inventory *inv);

  void addToCustomerHistory(Transaction *t);

  void addToOnLoan(Inventory *inv);

  void removeFromOnLoan(Inventory *inv);

  void addToAlreadyReturn(Inventory *inv);

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;
};

#endif