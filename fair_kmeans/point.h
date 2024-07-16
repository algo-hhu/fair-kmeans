#pragma once

#include <vector>
#include <cstddef>

// Small class for storing weighted points
class Point
{

private:
  uint weight;
  int color;
  int position = -1;
  int label = -1;
  std::vector<double> coordinates;

public:
  Point();
  Point(uint w, int c, const std::vector<double> &crd);
  Point(uint w, int c, int p, const std::vector<double> &crd);
  Point(uint w, int c, int p, int l, const std::vector<double> &crd);
  void join(Point p);
  std::string print() const;
  std::string printCoordinatesWithSpaces() const;
  const std::vector<double> &getCoordinatePointer() const { return coordinates; };
  size_t getDimension() const { return coordinates.size(); };
  uint getWeight() const { return weight; };
  int getColor() const { return color; };
  int getPosition() const { return position; };
  int getLabel() const { return label; };
  void setWeight(uint w) { weight = w; };
  void setPosition(int p) { position = p; };
  void setLabel(int l) { label = l; };
  void addPoint(const Point &p, bool safe_mode);
  void divideCoords(int factor);

  static double sd(const Point &p, const Point &q);
  static Point join(const Point &p, const Point &q);
  static Point pointAtUnweightedCentroid(const Point &p, const Point &q, uint w);
};
