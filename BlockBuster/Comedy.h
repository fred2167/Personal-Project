#ifndef COMEDY_H
#define COMEDY_H

#include "Movie.h"
#include "util.h"

class ComedyFactory : public InventoryFactory {
public:
  ComedyFactory();
  Inventory *create() override;
};

// a concrete sub-class of Movie and Inventory with self-registered factory
class Comedy : public Movie {

public:
  bool operator<(Inventory &rhs) override;
  bool operator>(Inventory &rhs) override;
  bool operator==(Inventory &rhs) override;

  istream &transactionRead(istream &is) override;

  ostream &transactionPrinter(ostream &os) const override;

private:
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;
};

#endif