#ifndef MEDIATYPE_H
#define MEDIATYPE_H

#include <cassert>
#include <iostream>
#include <map>
#include <string>

using namespace std;

// forward declare for MediaTypeFactory, fully define later
class MediaType;

// abstract base class factory.
// Sub-class of MediaType with factory should be a sub-class of this factory
class MediaTypeFactory {

public:
  virtual MediaType *create() const = 0;
};

/* a *base abstract* class for all Media Type, such as DVD with an *abstract*
   factory this class contains a static factory for all MediaType factories. map
   <identifier :string, MediaTypeFactory *> factories
*/
class MediaType {

  friend ostream &operator<<(ostream &os, const MediaType &m);
  friend istream &operator>>(istream &is, MediaType &m);

protected:
  int stock = 0;
  int onLoan = 0;
  //====================Virtual Function =============================
public:
  // read itself, superclass will call this in operator>>
  virtual istream &read(istream &is) = 0;

  // print yourself, superclass will call operator<< on this function
  virtual ostream &printer(ostream &os) const = 0;

  virtual ~MediaType() = default;
  //====================Getter/ Setter=================================

  // stock manipulation
  bool hasStock();
  bool borrowItem(); // decrement stock by 1
  bool returnItem(); // increment stock by 1

  void setStock(int stock);
  int getStock();

  //====================Factory=======================================
  /* this method ensure movieFactories is created before main
  and in time for resgester*/
  static map<string, MediaTypeFactory *> &getMediaTypeFactory();

  static void registerMediaTypeFactory(const string &type,
                                       MediaTypeFactory *factory);

  static bool hasMediaTypeFactory(const string &type);

  static MediaType *createMediaType(const string &type);
};

#endif