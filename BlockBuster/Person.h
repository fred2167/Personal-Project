#ifndef PERSON_H
#define PERSON_H

#include <iostream>
#include <map>
#include <string>

using namespace std;

// forward declare Person for PersonFactory, fully define later
class Person;

// abstract base class factory.
// Sub-class of person with factory should be a sub-class of this factory
class PersonFactory {

public:
  virtual Person *create() = 0;
};

/* a *base abstract* class with an abstract factory.
   this class contains a static factory for all person factories.
   map <identifier :string, PersonFactory *> factories
*/
class Person {

  friend ostream &operator<<(ostream &os, const Person &p);
  friend istream &operator>>(istream &is, Person &p);

protected:
  int id;
  string name;

private:
  // read itself, remember to set stock, superclass will call this in operator>>
  virtual istream &read(istream &is) = 0;

  // print yourself, superclass will call operator<< on this function
  virtual ostream &printer(ostream &os) const = 0;

public:
  virtual ~Person() = default;

  int getId();
  string getName();

  bool operator<(Person &rhs);
  bool operator>(Person &rhs);
  bool operator==(Person &rhs);

  //====================Factory=======================================

  /* this method ensure personFactories is created before main
  and in time for resgester*/
  static map<string, PersonFactory *> &getPersonFactories();

  static void registerPersonFactory(const string &type, PersonFactory *factory);

  static bool hasPersonFactory(const string &type);

  static Person *createPerson(const string &type);
};

#endif