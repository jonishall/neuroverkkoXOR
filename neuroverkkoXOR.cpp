//============================================================================
// Name        : neuroverkkoXOR.cpp
// Author      : joni
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <ctime>
using namespace std;
float sigma(float x){ 					// tavallinen sigmoid
	return (1/(1+pow(M_E,-x))  );
}
float dsigma(float x){					//ylemmän derivaatta
	float apu=pow(M_E,-x);
	return apu/pow(1+apu,2);
}
float loss(int x1, int x2, float o){   //loss-funktio
	return pow((float)(x1^x2)-sigma(o),2);
}
int main() {
	srand(time(nullptr));
	float w11=(double)rand()/(double)RAND_MAX;			//painot ja vakiot
	float w12=(double)rand()/(double)RAND_MAX;			//2x2x1 neuroverkolle
	float w21=(double)rand()/(double)RAND_MAX;
	float w22=(double)rand()/(double)RAND_MAX;
	float wa=(double)rand()/(double)RAND_MAX;
	float wb=(double)rand()/(double)RAND_MAX;
	float c1=(double)rand()/(double)RAND_MAX;
	float c2=(double)rand()/(double)RAND_MAX;
	float c3=(double)rand()/(double)RAND_MAX;
	for(int i=0;i<1000;i++){

		float l=0;
		float dw11=0;
		float dw12=0;
		float dw21=0;
		float dw22=0;
		float dwa=0;
		float dwb=0;
		float dc1=0;
		float dc2=0;
		float dc3=0;
		for(int x1=0;x1<2;x1++){
			for(int x2=0;x2<2;x2++){
				float o1=w11*x1+w21*x2+c1;					//hidden layer ylempi
				float o2=w12*x1+w22*x2+c2;					//hidden layer alempi
				float o=wa*sigma(o1)+wb*sigma(o2)+c3;		//output
				float doutput=+(float)(x1^x2)-sigma(o);
				dw11+=doutput*dsigma(o)*wa*dsigma(o1)*x1;   // partial loss / partial dw11
				dw21+=doutput*dsigma(o)*wa*dsigma(o1)*x2;
				dw12+=doutput*dsigma(o)*wb*dsigma(o2)*x1;
				dw22+=doutput*dsigma(o)*wb*dsigma(o2)*x2;
				dwa +=doutput*dsigma(o)*sigma(o1);
				dwb +=doutput*dsigma(o)*sigma(o2);
				dc1 +=doutput*dsigma(o)*wa*dsigma(o1);
				dc2 +=doutput*dsigma(o)*wb*dsigma(o2);
				dc3 +=doutput*dsigma(o);
				l   +=loss(x1,x2,o);

			}
		float lr=6;
		w11+=lr*dw11;
		w12+=lr*dw12;
		w21+=lr*dw21;
		w22+=lr*dw22;
		wa +=lr*dwa;
		wb +=lr*dwb;
		c1 +=lr*dc1;
		c2 +=lr*dc2;
		c3 +=lr*dc3;

		}

		cout << l<<endl;


	}


	return 0;
}

