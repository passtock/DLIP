#include "TU_DLIP.h"

#include <iostream>

// =============================
// Exercise 1 :: Define Function
// =============================

int sum(int val1, int val2)
{
	return val1 + val2;
}

// ====================================
// Exercise 2 :: Create a Class "MyNum"
// ====================================

MyNum::MyNum(int x1, int x2)
{
	val1 = x1;
	val2 = x2;
}

int MyNum::sum(void)
{
	return val1 + val2;
}

void MyNum::print(void)
{
	std::cout << "MyNum.val1 : " << val1 << std::endl;
	std::cout << "MyNum.val2 : " << val2 << std::endl;
	std::cout << "Sum : " << sum() << std::endl;
}

// ======================================================
// Exercise 3: Create two Class "MyNum" in proj_A, proj_B
// ======================================================
proj_A::MyNum::MyNum(int x1, int x2, int x3)
{
	val1 = x1;
	val2 = x2;
	val3 = x3;
}

int proj_A::MyNum::sum(void)
{
	return val1 + val2 + val3;
}

void proj_A::MyNum::print(void)
{
	std::cout << "MyNum.val1 : " << val1 << std::endl;
	std::cout << "MyNum.val2 : " << val2 << std::endl;
	std::cout << "MyNum.val3 : " << val3 << std::endl;
	std::cout << "Sum : " << sum() << std::endl;
}

proj_B::MyNum::MyNum(int x1, int x2, int x3)
{
	val1 = x1;
	val2 = x2;
	val3 = x3;
}

int proj_B::MyNum::sum(void)
{
	return val1 + val2 + val3;
}

void proj_B::MyNum::print(void)
{
	std::cout << "MyNum.val1 : " << val1 << std::endl;
	std::cout << "MyNum.val2 : " << val2 << std::endl;
	std::cout << "MyNum.val3 : " << val3 << std::endl;
	std::cout << "Sum : " << sum() << std::endl;
}