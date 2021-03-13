#ifndef DVD_H
#define DVD_H

#include "MediaType.h"
#include <iostream>

using namespace std;

class DVDFactory : public MediaTypeFactory {

public:
  DVDFactory();

  MediaType *create() const override;
};

class DVD : public MediaType {

public:
  // copy constructor as default
  DVD(const DVD &rhs) = default;

  // move not allowed
  DVD(DVD &&rhs) = delete;

  // assignment as default
  DVD &operator=(const DVD &rhs) = default;

  // move assignment not allowed
  DVD &operator=(const DVD &&rhs) = delete;

  DVD() = default;

  ~DVD() override = default;

  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;
};

#endif