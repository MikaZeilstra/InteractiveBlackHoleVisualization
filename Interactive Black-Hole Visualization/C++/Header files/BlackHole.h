#pragma once
#include "../../CUDA/Header files/Metric.cuh"

class BlackHole
{
public:
	double a;
	BlackHole(double afactor) {
		setA(afactor);
	};

	void setA(double afactor) {
		a = afactor;
		//integration::setA(a);
	}

	double getAngVel(double mass) {
		return mass*a;
	};
	~BlackHole(){};
};


