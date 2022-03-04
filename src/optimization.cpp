#include <iostream>
#include <vector>
#include <random>

#include "glog/logging.h"

#include "../include/optimization.h"
#include "../include/tensor.h"

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"

using std::cout;
using std::endl;
using std::vector;


struct ReprojectErrorExp {

	ReprojectErrorExp(const std::vector<double>& pose, int numLms, const std::vector<std::vector<cv::Point3f>>& blendshapes, std::vector<cv::Point2f>& gtLms) {
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
			
			T sumW = T(0);
			T w0;
			for (int j = 1; j < 47; j++) {
				X += T(_blendshapes[j][i].x) * w[j];
				Y += T(_blendshapes[j][i].y) * w[j];
				Z += T(_blendshapes[j][i].z) * w[j];
				sumW += w[j];
			}
			w0 = T(1.0) - sumW;
			X += T(_blendshapes[0][i].x) * w0;
			Y += T(_blendshapes[0][i].y) * w0;
			Z += T(_blendshapes[0][i].z) * w0;

			//================= transforming from object to camera coordinate system 
			T extrinsicsVec[6];
			for (int j = 0; j < 6; j++)
				extrinsicsVec[j] = T(_pose[j]);

			// rotation
			T vert[3] = { X, Y, Z };
			T rotatedVert[3];

			//given 1x6 we only use 1x3 in ceres rotation
			ceres::AngleAxisRotatePoint(extrinsicsVec, vert, rotatedVert);

			// translation
			rotatedVert[0] += extrinsicsVec[3];//X_cam
			rotatedVert[1] += extrinsicsVec[4];//Y_cam
			rotatedVert[2] += extrinsicsVec[5];//Z_cam

			//================= handling the residual block

			T xp = rotatedVert[0] / rotatedVert[2];          // X_cam / Z_cam (3D to 2D answers)
			T yp = rotatedVert[1] / rotatedVert[2];          // Y_cam / Z_cam (3D to 2D answers)

			residual[2 * i] = T(_gtLms[i].x) - xp;    // if you follows the steps above, you can see xp and yp are directly influenced by w, as if   
			residual[2 * i + 1] = T(_gtLms[i].y) - yp;    // you are optimizing the effect w_exp on xp and yp, and their yielded error.

		}
		return true;
	}

private:
	std::vector<std::vector<cv::Point3f>>			_blendshapes;
	std::vector<double>			_pose;
	int						_numLms;
	vector<cv::Point2f>     _gtLms;
};

struct Regularization {
	Regularization(int numWeights, const vector<double>& wr, double penalty) {
		_numWeights = numWeights;
		_wr = wr;
		_penalty = penalty;
	}
	template <typename T>
	bool operator()(const T* w, T* residual) const {

		for (int i = 0; i < _numWeights; i++)
		{
			residual[i] = T(_wr[i]) - w[i];
			residual[i] *= T(_penalty);
		}
		return true;
	}

private:
	int             _numWeights ;
	vector<double>  _wr;
	double          _penalty;
};

//lms = 2D landmarks used to construct x - cx / f
//pose = 1x6 vector with rotation and translation
//image = needed to get cx and cy
//f = focal length
//w_exp = output

//function to call after pose estimation to optimize the data
bool optimize(std::vector<std::vector<cv::Point3f>> multExp, const vector<cv::Point2f>& lms,
	const std::vector<double>& pose, const cv::Mat& image, float f, Eigen::VectorXf& w_exp)
{

	int numExpressions = 47;
	int numLms = lms.size();
	float cx = image.cols / 2.0;
	float cy = image.rows / 2.0;

	vector<double> w(numExpressions, 0);        //numExpressions = 47

	vector<double> wr(numExpressions, 0);		//for regualrization
	wr[22] = 1;

	ceres::Problem problem;

	//3D to 2D equations ((x/y) = (x/y)/z + f + c(x/y)) --> ((x/y) - c(x/y)) / f = (x/y)/z
	vector<cv::Point2f> gtLms;
	gtLms.reserve(numLms);
	for (int i = 0; i < numLms; i++) {
		float gtX = (lms[i].x - cx) / f;
		float gtY = (lms[i].y - cy) / f;
		gtLms.emplace_back(gtX, gtY);
	}

	ReprojectErrorExp* repErrFunc = new ReprojectErrorExp(pose, numLms, multExp, gtLms);   // upload the required parameters
	ceres::CostFunction* optimTerm = new ceres::AutoDiffCostFunction<ReprojectErrorExp, ceres::DYNAMIC, 47>(repErrFunc, numLms * 2);  // times 2 becase we have gtx and gty
	problem.AddResidualBlock(optimTerm, NULL, &w[0]);

	float penalty = 0.3;
	Regularization* regular = new Regularization(47, wr, penalty);
	//optimTerm = new ceres::AutoDiffCostFunction<Regularization, 46, 46>(regular);
	//problem.AddResidualBlock(optimTerm, NULL, &w[0]);

	for (int i = 1; i < numExpressions ; i++) {
		problem.SetParameterLowerBound(&w[0], i, 0.0);   // first argument must be w of ZERO and the second is the index of interest
		problem.SetParameterUpperBound(&w[0], i, 1.0);    // also the boundaries should be set after adding the residual block
	}

	ceres::Solver::Options options;
	options.max_num_iterations = 50;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	cout << summary.BriefReport() << endl << endl;
	double sumW = 0;
	for (int i = 1; i < numExpressions; i++)
	{
		w_exp(i) = w[i];
		cout << "i=" << i << ", w[i]= " << w[i] << ", wr[i]= " << wr[i] << endl;
		sumW += w[i];
	}
	w_exp(0) = 1 - sumW;
	cout << "i=" << 0 << ", w[i]= " << w[0] << ", wr[i]= " << wr[0] << endl;

	return summary.termination_type == ceres::TerminationType::CONVERGENCE;
}