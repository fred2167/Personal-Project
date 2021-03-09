#ifndef CUSTOMER_H
#define CUSTOMER_H

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
  ~Customer() override;

  void displayCutomerHistory();

  void addToCustomerHistory(Transaction *t);

  void addToOnLoan(Inventory *inv);

  void removeFromOnLoan(Inventory *inv);

  void addToAlreadyReturn(Inventory *inv);

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;
};

#endif