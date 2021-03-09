#ifndef MOVIE_H
#define MOVIE_H

#include "Inventory.h"

using namespace std;

/* an abstract class, does not implement virtual function and no factory
  Contain only shared attributes for all movies and getters/setters for the
  attributes
*/
class Movie : public Inventory {

protected:
  string director = "";
  string title = "";
  int yearRelease = 0;
};
#endif