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
  istream &read(istream &is) override;

  ostream &printer(ostream &os) const override;
};

#endif