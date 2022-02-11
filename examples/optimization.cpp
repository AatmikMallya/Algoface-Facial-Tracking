#include <iostream>
#include <vector>
#include <random>

#include "ceres/ceres.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

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

struct Exponential {

    Exponential(int numObservations, const vector<double>& x, const vector<double>& y) {
        _numObservations = numObservations;
        _x.resize(numObservations);
        _y.resize(numObservations);
        std::copy(x.begin(), x.end(), _x.begin());
        std::copy(y.begin(), y.end(), _y.begin());
    }

    template <typename T>
    bool operator()(const T* w, T* residual) const {

        for (int i = 0; i < _numObservations; i++)
            residual[i] = T(_y[i]) - (w[0] * (T(_x[i]) * T(_x[i])) + (w[1] * T(_x[i])) + w[2]);

        return true;
    }

private:
    int                 _numObservations = 0;
    vector<double>      _x;
    vector<double>      _y;
};

struct Expression {

    Expression(int numObservations, const vector<double>& x, const vector<double>& y) {
        _numObservations = numObservations;
        _x.resize(numObservations);
        _y.resize(numObservations);
        std::copy(x.begin(), x.end(), _x.begin());
        std::copy(y.begin(), y.end(), _y.begin());
    }

    template <typename T>
    bool operator()(const T* w, T* residual) const {

        for (int i = 0; i < _numObservations; i++)
            residual[i] = T(_y[i]) - (w[0] * (T(_x[i]) * T(_x[i])) + (w[1] * T(_x[i])) + w[2]);

        return true;
    }

private:
    int                 _numObservations = 0;
    vector<double>      _x;
    vector<double>      _y;
};


int main(int argc, char** argv) {

    // a = 2, b = 3, c = -2

    int numObservations = 50;
    vector<double> x(numObservations);
    vector<double> y(numObservations);
    for (int i = 0; i < numObservations; i++) {
        x[i] = rand() * 7.0 / RAND_MAX;
        //y[i] = 2 * x[i] - 3 + rand() * 1.0 / RAND_MAX;
        y[i] = (2 * (x[i] * x[i])) + (3 * x[i]) - 2;
    }

    vector<double> w = { 0, 0, 0 };
    ceres::Problem problem;
    Exponential* exp = new Exponential(numObservations, x, y);
    
	ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Exponential, ceres::DYNAMIC, 3>(exp, numObservations);
	//ceres::CostFunction* costFunction = new ceres::AutoDiffCostFunction<Linear, 20, 2>(lin);
	
    problem.AddResidualBlock(costFunction, NULL, &w[0]);
	
    //problem.SetParameterLowerBound(&w[0], 0, 0);  
    //problem.SetParameterUpperBound(&w[0], 0, 4);
	
    //problem.SetParameterLowerBound(&w[0], 1, -5); 
    //problem.SetParameterUpperBound(&w[0], 1, -1);

    problem.SetParameterLowerBound(&w[0], 2, -1.9); 
    problem.SetParameterUpperBound(&w[0], 2, -1);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl << endl;
    for (int i = 0; i < w.size(); i++)
        cout << "i = " << i << ", w = " << w[i] << endl;

    return 0;
}