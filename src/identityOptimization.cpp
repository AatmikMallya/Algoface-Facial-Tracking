#include <iostream>
#include <filesystem>
#include <vector>
#include <random>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "../include/identityOptimization.h"

#include "../include/tensor.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;


struct ReprojectErrorIdn {

	ReprojectErrorIdn(const std::vector<float>& pose, int numLms, const std::vector<std::vector<cv::Point3f>>& blendshapes, vector<cv::Point2f>& gtLms) {
		_pose = pose;
		_numLms = numLms;
		_blendshapes = blendshapes;
		_gtLms = gtLms;
	}

	template <typename T>
	bool operator()(const T* w, T* residual) const {

		for (int i = 0; i < _numLms; i++) {

			//============= linear combination
			T X = T(0);
			T Y = T(0);
			T Z = T(0);

			for (int j = 0; j < 150; j++) {
				X += T(_blendshapes[j][i].x) * w[j];
				Y += T(_blendshapes[j][i].y) * w[j];
				Z += T(_blendshapes[j][i].z) * w[j];
			}

			//================= transforming from object to camera coordinate system 
			T extrinsicsVec[6];
			for (int j = 0; j < 6; j++)
				extrinsicsVec[j] = T(_pose[j]);

			// rotation
			T vert[3] = { X, Y, Z };
			T rotatedVert[3];
			ceres::AngleAxisRotatePoint(extrinsicsVec, vert, rotatedVert);

			// translation
			rotatedVert[0] += extrinsicsVec[3];
			rotatedVert[1] += extrinsicsVec[4];
			rotatedVert[2] += extrinsicsVec[5];

			//================= handling the residual block

			T xp = rotatedVert[0] / rotatedVert[2];          // X_cam / Z_cam
			T yp = rotatedVert[1] / rotatedVert[2];          // Y_cam / Z_cam

			residual[2 * i] = T(_gtLms[i].x) - xp;    // if you follows the steps above, you can see xp and yp are directly influenced by w, as if   
			residual[2 * i + 1] = T(_gtLms[i].y) - yp;    // you are optimizing the effect w_exp on xp and yp, and their yielded error.
		}
		return true;
	}

private:
	std::vector<std::vector<cv::Point3f>>			_blendshapes;
	std::vector<float>					_pose;
	int						_numLms;
	vector<cv::Point2f>     _gtLms;
};

struct Regularization {
	Regularization(int numWeights, double penalty) {
		_numWeights = numWeights;
		_penalty = penalty;
	}
	template <typename T>
	bool operator()(const T* w, T* residual) const {

		T sum = T(0);
		for (int i = 0; i < _numWeights; i++)
		{
			sum += w[i] * w[i];
		}

		residual[0] = sum - T(1);
		residual[0] *= T(_penalty);

		return true;
	}

private:
	int             _numWeights = 0;
	double          _penalty;
};


bool identityOptimize(const vector<cv::Point2f>& lms,
	const std::vector<float>& pose, const cv::Mat& image, float f, Eigen::VectorXf& w_idn,
	const std::vector<std::vector<cv::Point3f>>& multIdn)
{

	int numIdentities = 150;
	int numLms = lms.size();
	float cx = image.cols / 2.0;
	float cy = image.rows / 2.0;

	vector<double> w(numIdentities, 1.0/150.0);
	

	ceres::Problem problem;

	vector<cv::Point2f> gtLms;
	gtLms.reserve(numLms);
	for (int i = 0; i < numLms; i++) {
		float gtX = (lms[i].x - cx) / f;
		float gtY = (lms[i].y - cy) / f;
		gtLms.emplace_back(gtX, gtY);
	}


	ReprojectErrorIdn* repErrFunc = new ReprojectErrorIdn(pose, numLms, multIdn, gtLms);   // upload the required parameters
	ceres::CostFunction* optimTerm = new ceres::AutoDiffCostFunction<ReprojectErrorIdn, ceres::DYNAMIC, 150>(repErrFunc, numLms * 2);  // times 2 becase we have gtx and gty
	problem.AddResidualBlock(optimTerm, NULL, &w[0]);

	for (int i = 0; i < numIdentities; i++) {
		problem.SetParameterLowerBound(&w[0], i, -1.0);   // first argument must be w of ZERO and the second is the index of interest
		problem.SetParameterUpperBound(&w[0], i, 1.0);    // also the boundaries should be set after adding the residual block
	}

	float penalty = 1.0;
	Regularization* regular = new Regularization(150, penalty);
	ceres::CostFunction* reg = new ceres::AutoDiffCostFunction<Regularization, 150, 150>(regular);
	problem.AddResidualBlock(reg, NULL, &w[0]);

	ceres::Solver::Options options;
	options.max_num_iterations = 35;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	cout << summary.BriefReport() << endl << endl;
	float sum = 0;
	for (int i = 0; i < numIdentities; i++)
	{
		w_idn(i) = w[i];
	}

	return summary.termination_type == ceres::TerminationType::CONVERGENCE;
}
