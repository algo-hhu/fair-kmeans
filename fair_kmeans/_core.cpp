#include <Python.h>
#include <iostream>
#include <cstdint>

#include "fair_clustering_tools.h"

typedef unsigned int uint;



bool checkWeightValidity(std::vector<int> &weightsum)
{
    int first_sum = weightsum[0];

    for (size_t i = 1; i < weightsum.size(); ++i)
    {
        if (weightsum[i] != first_sum)
        {
            std::cout << "Total weight sum of color 0 is " << first_sum << ", but weight sum of color " << i << " is " << weightsum[1] << ". All weights must be the same." << std::endl;
            return false;
        }
    }
    return true;
}

// Splits points into weight, color and coordinates.
// Processes the data into a n-color-dim array points,
// where each index is one color.
// Number and weightsum give the number
// and total weight of points in either array.
void postprocessData(double *array, double *sampleWeights, int *colors, uint n, uint d, std::vector<std::vector<Point>> &points, std::vector<int> &number, std::vector<int> &weightsum)
{
    int j = 0;
    for (uint i = 0; i < n * d; i += d)
    {
        // In a future version, we might consider using weights as doubles
        // but currently the algorithm does not allow for that
        int w = static_cast<int>(sampleWeights[j]);
        int c = colors[j];
        number[c] += 1;
        weightsum[c] += w;
        std::vector<double> coords(&array[i], &array[i] + d);
        // Add j as the position of the point in the array
        points[c].push_back(Point(w, c, j, coords));
        j++;
    }
}

// Convert the array of centers to a vector of points
// Postprocess the center data into a vector of Point objects
// centers have weight 1 and no color (-1).
void postprocessCenters(double *centers, std::vector<Point> &centersVector, uint k, uint d)
{
    // Create center vector with points objects by copying from centerdata
    int j = 0;
    for (uint i = 0; i < k * d; i += d)
    {
        std::vector<double> coords(&centers[i], &centers[i] + d);
        // Centers have weight 1 and no color (-1)
        centersVector[j] = Point(1, -1, coords);
        j++;
    }
}

// Copy the centers back to the centers array to return to Python
void copyBackCenters(double *centers, std::vector<Point> &centersVector)
{
    for (size_t i = 0; i < centersVector.size(); ++i)
    {
        size_t d = centersVector[i].getDimension();
        for (size_t j = 0; j < d; ++j)
        {
            centers[i * d + j] = centersVector[i].getCoordinatePointer()[j];
        }
    }
}

void assignLabels(std::vector<std::vector<Point>> &points, int *labels)
{
    for (size_t c = 0; c < points.size(); ++c)
    {
        for (size_t j = 0; j < points[c].size(); ++j)
        {
            labels[points[c][j].getPosition()] = points[c][j].getLabel();
        }
    }
}


// Thank you https://github.com/dstein64/kmeans1d!

extern "C"
{
// "__declspec(dllexport)" causes the function to be exported when compiling on Windows.
// Otherwise, the function is not exported and the code raises
//   "AttributeError: function 'cluster' not found".
// Exporting is a Windows platform requirement, not just a Visual Studio requirement
// (https://stackoverflow.com/a/22288874/1509433). The _WIN32 macro covers the Visual
// Studio compiler (MSVC) and MinGW. The __CYGWIN__ macro covers gcc and clang under
// Cygwin.
#if defined(_WIN32) || defined(__CYGWIN__)
    __declspec(dllexport)
#endif
    double
    fairKMeans(double *array,
               double *sampleWeights,
               int *colors,
               uint n,
               uint d,
               uint k,
               uint nc,
               uint maxIterations,
               double tolerance,
               int seed,
               int *labels,
               double *centers,
               bool updateCenters,
               uint *iterations)
    {

        std::vector<int> weightsum(nc, 0);
        std::vector<int> number(nc, 0);
        // One vector of points per color
        std::vector<std::vector<Point>> points(nc);

        postprocessData(array, sampleWeights, colors, n, d, points, number, weightsum);

        // std::cout << "Points:" << std::endl;
        // for (size_t c = 0; c < nc; ++c)
        // {
        //     for (size_t i = 0; i < points[c].size(); ++i)
        //     {
        //         std::cout << points[c][i].print() << std::endl;
        //     }
        // }

        std::vector<Point> centersVector(k);
        postprocessCenters(centers, centersVector, k, d);

        if (!checkWeightValidity(weightsum))
        {
            return -1;
        }

        // std::cout << "Initial Centers:" << std::endl;
        // for (size_t i = 0; i < centersVector.size(); ++i)
        // {
        //     std::cout << centersVector[i].print() << std::endl;
        // }

        double cost = std::numeric_limits<double>::max();
        double previousCost = cost;
        bool change = true;

        while (*iterations < maxIterations && change)
        {

            // Computing the fair assignment and better centers
            // If we don't want to update the centers, then we shoudn't fill all clusters
            CentersValueChange result = compute_fair_assignment(points, centersVector, number, weightsum, updateCenters, seed);

            // If the returned cost is negative, the CapacityScaling algorithm failed
            if (std::get<1>(result) < 0.0)
            {
                return std::get<1>(result);
            }

            std::vector<Point> &newCenters = std::get<0>(result);

            // std::cout << std::endl;
            // std::cout << "New Centers:" << std::endl;
            // for (size_t i = 0; i < newCenters.size(); ++i)
            // {
            //     std::cout << "Center " << i << ": " << newCenters[i].print() << std::endl;
            // }

            centersVector = newCenters;
            cost = std::get<1>(result);


            // std::cout << "Cost: " << cost << std::endl;
            // std::cout << "Previous Cost: " << previousCost << std::endl;

            ++*iterations;
            change = false;

            // If we don't see any improvement anymore, we stop
            if (cost < (1 - tolerance) * previousCost)
            {
                previousCost = cost;
                change = true;
            }
        }

        if (cost >= 0)
        {
            if (updateCenters) copyBackCenters(centers, centersVector);

            assignLabels(points, labels);
        }

        return cost;
    }
} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "fair_kmeans._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void)
{
    return PyModule_Create(&_coremodule);
}
