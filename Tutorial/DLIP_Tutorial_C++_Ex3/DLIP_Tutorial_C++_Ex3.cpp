#include "../../Include/TU_DLIP.h"

void main()
{
	proj_A::MyNum mynum1(1, 2, 3);
	proj_B::MyNum mynum2(4, 5, 6);

	mynum1.print();
	mynum2.print();

	system("pause");
}