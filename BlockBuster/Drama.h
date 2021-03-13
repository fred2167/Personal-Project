#ifndef DRAMA_H
#define DRAMA_H

#include "Movie.h"
#include "util.h"

class DramaFactory : public InventoryFactory {
public:
  DramaFactory();
  Inventory *create() override;
};

// a concrete sub-class of Movie and Inventory with self-registered factory
class Drama : public Movie {

public:
  // copy constructor as default
  Drama(const Drama &rhs) = default;

  // move not allowed
  Drama(Drama &&rhs) = delete;

  // assignment as default
  Drama &operator=(const Drama &rhs) = default;

  // move assignment not allowed
  Drama &operator=(const Drama &&rhs) = delete;

  Drama();

  ~Drama() override = default;

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