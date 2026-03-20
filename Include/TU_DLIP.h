#ifndef _TU_DLIP_H		// same as "#if !define _TU_DLIP_H" (or #pragma once) 
#define _TU_DLIP_H

#include <iostream>

// =============================
// Exercise 1: Define Function
// =============================

int sum(int val1, int val2);

// ====================================
// Exercise 2: Create a Class "MyNum"
// ====================================

class MyNum
{
public:
	MyNum(int x1, int x2);
	int val1;
	int val2;
	int sum(void);
	void print(void);
};

// ======================================================
// Exercise 3: Create two Class "MyNum" in proj_A, proj_B
// ======================================================
namespace proj_A
{
	class MyNum
	{
	public:
		MyNum(int x1, int x2, int x3);
		int val1;
		int val2;
		int val3;
		int sum(void);
		void print(void);
	};
}

namespace proj_B
{
	class MyNum
	{
	public:
		MyNum(int x1, int x2, int x3);
		int val1;
		int val2;
		int val3;
		int sum(void);
		void print(void);
	};
}
#endif // !_TU_DLIP_H