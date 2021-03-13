#ifndef CLASSICS_H
#define CLASSICS_H

#include "Movie.h"
#include "util.h"

class ClassicsFactory : public InventoryFactory {
public:
  ClassicsFactory();
  Inventory *create() override;
};

// a concrete sub-class of Movie and Inventory with self-registered factory
class Classics : public Movie {

protected:
  int monthRelease = 0;
  string majorActor = "";

public:
  // copy constructor as default
  Classics(const Classics &rhs) = default;
  // move not allowed
  Classics(Classics &&rhs) = delete;

  // assignment as default
  Classics &operator=(const Classics &rhs) = default;

  // move assignment not allowed
  Classics &operator=(const Classics &&rhs) = delete;

  Classics();

  ~Classics() override = default;

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