//#include "TU_DLIP.h"
#include "../../Include/TU_DLIP.h"

int main()
{
	// =============================
	// Exercise 1: Define Function
	// =============================

	int val1 = 11;
	int val2 = 22;

	int out = sum(val1, val2);

	std::cout << out << std::endl;

	// ====================================
	// Exercise 2: Create a Class 'MyNum'
	// ====================================

	MyNum mynum(10, 20);
	mynum.print();

}