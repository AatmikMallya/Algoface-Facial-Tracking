#include <iostream>
#include <vector>
#include <random>

#include "ceres/ceres.h"

using std::cout;
using std::endl;
using std::vector;

struct Linear {

    Linear(int numObservations, const vector<double>& x, const vector<double>& y) {
        _numObservations = numObservations;
        _x.resize(numObservations);
        _y.resize(numObservations);
        std::copy(x.begin(), x.end(), _x.begin());
        std::copy(y.begin(), y.end(), _y.begin());
    }

    template <typename T>
    bool operator()(const T* w, T* residual) const {

        for (int i = 0; i < _numObservations; i++)
            residual[i] = T(_y[i]) - (w[0] * T(_x[i]) + w[1]);

        return true;
    }

private:
    int                 _numObservations = 0;
    vector<double>      _x;
    vector<double>      _y;
};

struct Regularization {
    Regularization(int numWeights, const vector<double>& wr, double penalty) {
        _numWeights = numWeights;
        _wr = wr;
        _penalty = penalty;
    }
    template <typename T>
    bool operator()(const T* w, T* residual) const {

        residual[0] = T(_wr[0]) - w[0];
        residual[1] = T(_wr[1]) - w[1];

        residual[0] *= T(_penalty);
        residual[1] *= T(_penalty);

        return true;
    }

private:
    int             _numWeights = 0;
    vector<double>  _wr;
    double          _penalty;
};

int main(int argc, char** argv) {

    int numObservations = 20;

    vector<double> x(numObservations);
    vector<double> y(numObservations);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, INT_MAX);
    for (int i = 0; i < numObservations; i++) {
        x[i] = rand() * 7.0 / RAND_MAX;
        //y[i] = 2 * x[i] - 3 + rand() * 1.0 / RAND_MAX;
        y[i] = 2 * x[i] - 3;
    }

    vector<double> w = { 0, 0 };
    ceres::Problem problem;
    Linear* lin = new Linear(numObservations, x, y);

    ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Linear, ceres::DYNAMIC, 2>(lin, numObservations);
    problem.AddResidualBlock(costFunction, NULL, &w[0]);
    problem.SetParameterLowerBound(&w[0], 0, 0);
    problem.SetParameterUpperBound(&w[0], 0, 4);
    problem.SetParameterLowerBound(&w[0], 1, -5);
    problem.SetParameterUpperBound(&w[0], 1, -1);

    vector<double> wr = { 1.7, -3.3 };
    float penalty = 1.0;
    Regularization* regular = new Regularization(2, wr, penalty);
    costFunction = new ceres::AutoDiffCostFunction<Regularization, 2, 2>(regular);
    problem.AddResidualBlock(costFunction, NULL, &w[0]);

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl << endl;
    for (int i = 0; i < w.size(); i++)
        cout << "i = " << i << ", w = " << w[i] << endl;

    return 0;
}