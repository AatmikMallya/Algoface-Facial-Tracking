#include <iostream>
#include <vector>
#include <random>

#include "glog/logging.h"

#include "../include/identityOptimization.h"
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


struct ReprojectErrorId {

	ReprojectErrorId(const std::vector<double>& pose, int numLms, const std::vector<std::vector<cv::Point3f>>& blendshapes, std::vector<cv::Point2f>& gtLms) {
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

			//given 1x6 we only use 1x3 in ceres rotation
			ceres::AngleAxisRotatePoint(extrinsicsVec, vert, rotatedVert);

			// translation
			rotatedVert[0] += extrinsicsVec[3];
			rotatedVert[1] += extrinsicsVec[4];
			rotatedVert[2] += extrinsicsVec[5];

			//================= handling the residual block

			T xp = rotatedVert[0] / rotatedVert[2];          // X_cam / Z_cam (3D to 2D answers)
			T yp = rotatedVert[1] / rotatedVert[2];          // Y_cam / Z_cam (3D to 2D answers)

			residual[2 * i] = T(_gtLms[i].x) - xp;    // if you follows the steps above, you can see xp and yp are directly influenced by w, as if   
			residual[2 * i + 1] = T(_gtLms[i].y) - yp;    // you are optimizing the effect w_exp on xp and yp, and their yielded error.
			//cout << "program got here "<< i << endl;
		}
		//cout << "program got here" << endl;
		return true;
	}

private:
	std::vector<std::vector<cv::Point3f>>			_blendshapes;
	std::vector<double>							_pose;
	int										_numLms;
	vector<cv::Point2f>					 _gtLms;
};

struct Regularization {

	Regularization(int numWeights, const vector<double>& wr, double penalty) {
		_numWeights = numWeights;
		_wr = wr;
		_penalty = penalty;
	}

	template <typename T>
	bool operator()(const T* w, T* residual) const {

		for (int i = 0; i < _numWeights; i++) {

			residual[i] = T(_wr[i]) - w[i];
			residual[i] *= T(_penalty);
		}
		//cout << "program got here2" << endl;
		return true;
	}

private:
	int							_numWeights = 0;
	std::vector<double>			_wr;
	double						_penalty;
};

//lms = 2D landmarks used to construct x - cx / f
//pose = 1x6 vector with rotation and translation
//image = needed to get cx and cy
//f = focal length
//w_exp = output

//function to call after pose estimation to optimize the data
bool identityOptimize(std::vector<std::vector<cv::Point3f>> multIdn, const vector<cv::Point2f>& lms,
	const std::vector<double>& pose, const cv::Mat& image, double f, Eigen::VectorXf& w_idn)
{
	/*
	//3D slice of matrix
	Eigen::MatrixXf sparseBlendshapes;
	for (int i = 0; i < 47; i++)
	{
		for (int j = 0; j < 73; j++)
		{
			sparseBlendshapes = shapeTensor(137, i, j);
		}
	}
	*/

	int numIdentity = 150;
	int numLms = lms.size(); //73
	double cx = image.cols / 2.0;
	double cy = image.rows / 2.0;

	vector<double> w(numIdentity, 0);        // numExpressions = 47
	w[137] = 1;

	vector<double> wr(numIdentity, 0);
	//wr[137] = 1; //changes the face identity


	ceres::Problem problem;

	//3D to 2D equations ((x/y) = (x/y)/z + f + c(x/y)) --> ((x/y) - c(x/y)) / f = (x/y)/z
	vector<cv::Point2f> gtLms;
	gtLms.reserve(numLms);
	for (int i = 0; i < numLms; i++) {
		double gtX = (lms[i].x - cx) / f;
		double gtY = (lms[i].y - cy) / f;
		gtLms.emplace_back(gtX, gtY);
	}

	//identity linear combination / optimization
	ReprojectErrorId* repErrFunc = new ReprojectErrorId(pose, numLms, multIdn, gtLms);
	ceres::CostFunction* optimTerm = new ceres::AutoDiffCostFunction<ReprojectErrorId, ceres::DYNAMIC, 150>(repErrFunc, numLms * 2);
	problem.AddResidualBlock(optimTerm, NULL, &w[0]);

	for (int i = 0; i < numIdentity; i++) {
		problem.SetParameterLowerBound(&w[0], i, 0.0);   // first argument must be w of ZERO and the second is the index of interest
		problem.SetParameterUpperBound(&w[0], i, 1.0);    // also the boundaries should be set after adding the residual block
	}

	//regularization for identity optimization
	float penalty = 1.0;
	Regularization* regular = new Regularization(150, wr, penalty); //numIdentity
	optimTerm = new ceres::AutoDiffCostFunction<Regularization, 150, 150>(regular);
	problem.AddResidualBlock(optimTerm, NULL, &w[0]);

	//Error = proj + regularizatiopn(penalty)

	//cout << "program got here" << endl;

	ceres::Solver::Options options;
	//options.logging_type = ceres::SILENT;
	options.max_num_iterations = 50;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	cout << summary.BriefReport() << endl << endl;

	for (int i = 0; i < numIdentity; i++)
	{
		w_idn(i) = w[i];
	}
	//cout << "program got here" << endl;
	//for (int i = 0; i < w.size(); i++)
	//{
	//	cout << "i = " << i << " w = " << w[i] << endl;
	//}
	//w_exp.all() = w.data;
	//cout << "before return" << endl;
	return summary.termination_type == ceres::TerminationType::CONVERGENCE;
	//return true;
	//return false;

}

