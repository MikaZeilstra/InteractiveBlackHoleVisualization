#pragma once
#include "../../CUDA/Header files/Metric.cuh"


class Camera
{
public:
	double theta, phi, r; 
	double speed, br, btheta, bphi;

	double alpha, w, wbar, Delta, ro;

	Camera(){};

	//Camera(Camera &c){
	//	theta = c.theta;
	//	phi = c.phi;
	//	r = c.r;
	//	speed = c.speed;
	//	alpha = c.alpha;
	//	w = c.w;
	//	wbar = c.wbar;
	//	Delta = c.Delta;
	//	ro = c.ro;
	//};

	Camera(double theCam, double phiCam, double radfactor, double speedCam)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;
		speed = speedCam;

		bphi = 1;
		btheta = 0;
		br = 0;
		initforms();
	};

	Camera(double theCam, double phiCam, double radfactor, double _br, double _btheta, double _bphi)
	{
		theta = theCam;
		phi = phiCam;
		r = radfactor;

		bphi = _bphi;
		btheta = _btheta;
		br = _br;

		speed = metric::calcSpeed(r, theta);
		initforms();
	};

	/// <summary>
	/// Camera constructor for given location speed and direction
	/// </summary>
	/// <param name="cam_pos"> Camera positon in spherical coordinates (R, Theta,Phi) </param>
	/// <param name="cam_speed_dir">Cameras direction of movement in spherical coordinates</param>
	/// <param name="_speed"> Camera speed</param>
	Camera(double3 cam_pos, double3 cam_speed_dir, double _speed)
	{
		theta = cam_pos.y;
		phi = cam_pos.z;
		r = cam_pos.x;

		bphi = cam_speed_dir.z;
		btheta = cam_speed_dir.y;
		br = cam_speed_dir.x;

		speed = _speed;
		initforms();
	};


	void initforms() {
		alpha = metric::_alpha(this->r, this->theta,metric::sq(this->r),metric::sq(sin(theta)), metric::sq(cos(theta)));
		w = metric::_w(this->r, this->theta, metric::sq(this->r), metric::sq(sin(theta)));
		wbar = metric::_wbar(this->r, this->theta, metric::sq(this->r), metric::sq(sin(theta)), metric::sq(cos(theta)));
		Delta = metric::_Delta(this->r, metric::sq(this->r));
		ro = metric::_ro(this->r, this->theta, metric::sq(this->r), metric::sq(cos(theta)));

	};

	double getDistFromBH(float mass) {
		return mass*r;
	};

	std::vector<float> getParamArray() {
		std::vector<float> camera(10);
		camera[0] = speed;
		camera[1] = alpha;
		camera[2] = w;
		camera[3] = wbar;
		camera[4] = br;
		camera[5] = btheta;
		camera[6] = bphi;
		camera[7] = theta;
		camera[8] = phi;
		camera[9] = r;

		return camera;
	};

	~Camera(){};
};


