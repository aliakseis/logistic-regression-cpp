// logistic-regression-cpp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iterator>

#include <future>


/*

http://yann.lecun.com/exdb/mnist/

TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.

TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  10000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel

*/

namespace {

enum { DIM = 28 * 28 };

//typedef unsigned char AttributeType;

struct ObjectInfo
{
    double pos[DIM];
    int data;
};

typedef std::vector<ObjectInfo> ObjectInfos;

std::istream& operator %(std::istream& s, int32_t& v)
{
    s.read((char*)&v, sizeof(v));
    std::reverse((char*)&v, (char*)(&v + 1));
    return s;
}

ObjectInfos ReadDataSet(const char* imageFile, const char* labelFile)
{
    std::ifstream ifsImages(imageFile, std::ifstream::in | std::ifstream::binary);
    int32_t magic;
    ifsImages % magic;
    int32_t numImages;
    ifsImages % numImages;
    int32_t numRows, numCols;
    ifsImages % numRows % numCols;

    std::ifstream ifsLabels(labelFile, std::ifstream::in | std::ifstream::binary);
    ifsLabels % magic;
    int32_t numLabels;
    ifsLabels % numLabels;

    ObjectInfos infos;
    infos.resize(numImages);
    for (int i = 0; i < numImages; ++i)
    {
        unsigned char buffer[DIM];
        ifsImages.read((char*)buffer, DIM);
        for (int j = 0; j < DIM; ++j)
            infos[i].pos[j] = double(buffer[j]) / 255.;
        unsigned char label;
        ifsLabels.read((char*)&label, 1);
        infos[i].data = label;
    }

    const bool ok = ifsImages && ifsLabels;
    //const bool eof = ifsImages.eof() && ifsLabels.eof();

    return infos;
}

int sign(double v)
{
    return (v > 0) ? 1 : ((v < 0) ? -1 : 0);
}

double sigmoid(double z)
{
    return 1.0 / (1.0 + exp(-z));
}

} // namespace

int main()
{
    enum { N_LABELS = 10 };

    try
    {
        const auto trainingSet = ReadDataSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

        std::vector<double> results[10];

        std::vector<std::future<void>> futures;

        for (int label = 0; label < N_LABELS; ++label)
        {
            auto lam = [&trainingSet, label, &results]
            {
                enum { N_ITER = 120 };

                const double lambda = .015;
                const double lr = .97;

                std::vector<double> w(DIM + 1, -1.);

                //double prev_cost = DBL_MAX;

                for (int i = 1; i <= N_ITER; ++i)
                {
                    //double cost = 0.;

                    std::vector<double> delta_l(DIM + 1, 0.);
                    for (auto& v : trainingSet)
                    {
                        const int y = (v.data % 10 == label);

                        const auto& l = v.pos;
                        const auto h = sigmoid(std::inner_product(std::begin(l), std::end(l), w.begin(), *w.rbegin()));
                        const auto delta = h - y;

                        //cost += ((-y)*log(h) - (1-y)*log(1 - h)) / trainingSet.size();

                        for (int j = 0; j < delta_l.size() - 1; ++j)
                            delta_l[j] += l[j] * delta;

                        delta_l[delta_l.size() - 1] += delta;
                    }

                    //if (cost >= prev_cost)
                    //    break;
                    //prev_cost = cost;

                    for (int j = 0; j < delta_l.size(); ++j)
                    {
                        const auto delta = delta_l[j] / trainingSet.size() + lambda * w[j];
                        w[j] -= lr * delta;
                    }
                }

                results[label] = std::move(w);
            };
            futures.push_back(std::async(std::launch::async, lam));
        }

        futures.clear(); // waiting in destructors

        auto testSet = ReadDataSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

        int numMismatches = 0;

        for (auto& v : testSet)
        {
            double h_max = 0;
            int predicted = 0;
            const auto& l = v.pos;
            for (int label = 0; label < N_LABELS; ++label)
            {
                const auto& w = results[label];
                const auto h = sigmoid(std::inner_product(std::begin(l), std::end(l), w.begin(), *w.rbegin()));
                if (h > h_max)
                {
                    h_max = h;
                    predicted = label;
                }
            }
            if (predicted != (v.data % 10))
                ++numMismatches;
        }

        std::cout << "Test cases: " << testSet.size() << "; mismatches: " << numMismatches << '\n';
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Fatal: " << ex.what() << '\n';
    }
}
