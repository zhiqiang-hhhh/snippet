// C++ Program to evaluate the performance of the four
// execution policies
#include <chrono>
#include <execution>
#include <iostream>
#include <vector>

// Function to calculate the execution time of different
// execution policies
template<typename POLICY_TYPE>
void execTime(POLICY_TYPE policy_type, std::vector<int>& num,
			std::string pType_name)
{
	auto start_time
		= std::chrono::high_resolution_clock::now();

	long long sum = 0;

	// finding sum of each element in the vector
	std::for_each(policy_type, num.begin(), num.end(),
				[&](int n) { sum += n; });

	auto end_time
		= std::chrono::high_resolution_clock::now();

	auto taken_time = std::chrono::duration_cast<
						std::chrono::milliseconds>(
						end_time - start_time)
						.count();
	// printing execution time
	std::cout << pType_name
			<< " execution time: " << taken_time
			<< "ms\n";
}

int main()
{
	// Creating large vector of int
	int size = 9999999;
	std::vector<int> num(size);
	// initializing vector
	for (int i = 0; i < size; i++) {
		num[i] = i;
	}

	// execution time
	execTime(std::execution::seq, num, "Sequenced");
	execTime(std::execution::unseq, num, "Unsequenced");
	execTime(std::execution::par, num, "Parallel");
	execTime(std::execution::par_unseq, num,
			"Parallel Unsequenced");

	return 0;
}
