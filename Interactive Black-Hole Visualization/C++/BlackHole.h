#pragma once
#include "../CUDA/Metric.cuh"
#include "../CUDA/integration.cuh"

class BlackHole
{
public:
	double a;
	BlackHole(double afactor) {
		setA(afactor);
	};

	void setA(double afactor) {
		a = afactor;
		metric::setAngVel(afactor);
		//integration::setA(a);
	}

	double getAngVel(double mass) {
		return mass*a;
	};
	~BlackHole(){};
};


