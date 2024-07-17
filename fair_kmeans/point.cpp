#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstddef>

#include "point.h"

typedef unsigned int uint;

// print a point
std::string Point::print() const
{
  std::stringstream tmp;
  tmp << "(" << coordinates[0];
  for (uint i = 1; i < coordinates.size(); i++)
  {
    tmp << "," << coordinates[i];
  }
  tmp << "), weight: " << weight;
  tmp << ", color: " << color;
  return tmp.str();
}

// prints the coordinates into a string, separated by white spaces
std::string Point::printCoordinatesWithSpaces() const
{
  std::stringstream tmp;
  tmp << coordinates[0];
  for (uint i = 1; i < coordinates.size(); i++)
  {
    tmp << " " << coordinates[i];
  }
  return tmp.str();
}

// empty point constructor
Point::Point() : weight(0), color(0)
{
}

// standard point constructor
Point::Point(uint w, int c, const std::vector<double> &crd) : weight(w), color(c), coordinates(crd)
{
}

Point::Point(uint w, int c, int p, const std::vector<double> &crd) : weight(w), color(c), position(p), coordinates(crd)
{
}

Point::Point(uint w, int c, int p, int l, const std::vector<double> &crd) : weight(w), color(c), position(p), label(l), coordinates(crd)
{
}

// adds the point p to this point
void Point::addPoint(const Point &p, bool safe_mode)
{
  if (safe_mode)
  {
    std::cout << "safe mode is not implemented";
  }
  // weight of p is added
  weight += p.getWeight();
  const std::vector<double> &pcords = p.getCoordinatePointer();
  if (pcords.size() != coordinates.size())
  {
    throw std::out_of_range("point dimensions do not match");
  }
  // Coordinates of p are added to this points coordinates
  for (size_t i = 0; i < pcords.size(); ++i)
  {
    coordinates[i] += pcords[i];
  }
  // If the colors do not match then we set the color attribute to -1 for invalid.
  if (p.getColor() != color)
  {
    color = -1;
  }
}

// divides all coordinates by <factor>
void Point::divideCoords(int factor)
{
  for (size_t i = 0; i < coordinates.size(); ++i)
  {
    coordinates[i] = coordinates[i] / ((double)factor);
  }
}

// computes the squared distance between point p and point q
// weight and color have no influence
double Point::sd(const Point &p, const Point &q)
{
  // std::cout << p.print() <<std::endl;
  // std::cout << q.print()<<std::endl;
  const std::vector<double> &cp = p.getCoordinatePointer();
  const std::vector<double> &cq = q.getCoordinatePointer();
  if (p.coordinates.size() != q.coordinates.size())
  {
    std::cout << "p: " << p.print() << std::endl;
    std::cout << "q: " << q.print() << std::endl;
    throw std::range_error("Points of different lengths???");
  }
  double dist = 0;
  for (uint i = 0; i < cp.size(); i++)
  {
    //    std::cout << i<<": " <<(cp[i]-cq[i])<<std::endl;
    dist += (cp[i] - cq[i]) * (cp[i] - cq[i]);
  }
  return dist;
}

// Creates a point which is the centroid of p and q (weighted).
// Attention: The color of the joined point is set to -1.
// It really should be in [0,1], but I do not want to change
// the color attribute to a floating point.
Point Point::join(const Point &p, const Point &q)
{
  uint wp = p.getWeight();
  uint wq = q.getWeight();
  const std::vector<double> &coordsp = p.getCoordinatePointer();
  const std::vector<double> &coordsq = q.getCoordinatePointer();
  std::vector<double> coordssum(coordsp.size());
  for (size_t i = 0; i < coordsp.size(); i++)
  {
    coordssum[i] = (wp * coordsp[i] + wq * coordsq[i]) / ((double)(wp + wq));
  }
  return Point(wp + wq, -1, coordssum);
}

// Creates a point at the unweighted centroid of p and q
// and gives the point weight w.
// Color is -1 (invalid) since it would really be a
// fractional number and I don't want to support this.
Point Point::pointAtUnweightedCentroid(const Point &p, const Point &q, uint w)
{
  const std::vector<double> &coordsp = p.getCoordinatePointer();
  const std::vector<double> &coordsq = q.getCoordinatePointer();
  std::vector<double> coordssum(coordsp.size());
  for (size_t i = 0; i < coordsp.size(); i++)
  {
    coordssum[i] = (coordsp[i] + coordsq[i]) / (2.0);
  }
  return Point(w, -1, coordssum);
}
