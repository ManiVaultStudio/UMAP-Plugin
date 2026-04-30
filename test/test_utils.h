#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <vector>
#include <cstdint>
#include <utility>

namespace testing {

	/// /////// ///
	/// TESTING ///
	/// /////// ///

	// Test helper that compares two values with tolerance
	void expectNear(float expected, float actual, float tolerance = 1e-5f,
		const std::string& test_name = "") {
		if (std::fabs(expected - actual) > tolerance) {
			std::cerr << "Test failed: " << test_name
				<< "\n  Expected: " << expected
				<< "\n  Actual: " << actual
				<< "\n  Diff: " << std::fabs(expected - actual)
				<< "\n  Tolerance: " << tolerance << std::endl;
		}
		CHECK(expected == Catch::Approx(actual).margin(tolerance));
	}

	/// ///////////// ///
	/// DATA CREATION ///
	/// ///////////// ///

	template<typename scalar = float>
	class DataGenerator {
	public:
		DataGenerator(const scalar rangeMin = -1.f, const scalar rangeMax = 1.f) {
			gen.seed(42);
			dist = std::uniform_real_distribution<scalar>(rangeMin, rangeMax);
		}

		// Helper function to generate random vector
		std::vector<scalar> randomVector(const size_t dim) {
			std::vector<scalar> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = dist(gen);
			}
			return vec;
		}

		// Helper function to generate random matrix
		std::vector<scalar> randomMatrix(const size_t dim, const size_t numPoints) {
			return randomVector(dim * numPoints);
		}

		// Helper function to generate constant vector
		std::vector<scalar> constantVector(const size_t dim, const scalar value) {
			return std::vector<scalar>(dim, value);
		}

		// Helper function to generate linearly increasing vector
		std::vector<scalar> linearVector(const size_t dim, const scalar start = 0.0f, const scalar step = 1.0f) {
			std::vector<scalar> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = start + i * step;
			}
			return vec;
		}

	private:
		std::mt19937 gen;
		std::uniform_real_distribution<scalar> dist;
	};

	/// /////// ///
	/// LOGGING ///
	/// /////// ///

	template<typename T, typename S>
	inline void print(const std::pair<T, S>& pair) {
		std::cout << pair.first << " : " << pair.second << "\n";
	}

	template<typename T, typename S>
	inline void print(const std::vector<std::pair<T, S>>& pairs) {
		for (const auto& pair : pairs)
			print(pair);
		std::cout << std::endl;
	}

	template<typename T>
	void print(const std::vector<T>& vec) {
		for (const auto& val : vec) {
			std::cout << val << " ";
		}
		std::cout << std::endl;
	}

	inline void info(const std::string& message) {
		std::cout << message << std::endl;
	}

	inline void printDuration(const uint64_t t, const std::string& unit = "ms") {
		std::cout << "Duration: " << t <<  unit << std::endl;
	}

}
