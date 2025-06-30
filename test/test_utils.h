#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <iostream>
#include <cmath>
#include <random>
#include <vector>

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

	class DataGenerator {
	public:
		DataGenerator(const float range = 1.f) {
			gen.seed(42);
			dist = std::uniform_real_distribution<float>(-1.f * range, range);
		}

		// Helper function to generate random vector
		std::vector<float> randomVector(size_t dim) {
			std::vector<float> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = dist(gen);
			}
			return vec;
		}

		// Helper function to generate constant vector
		std::vector<float> constantVector(size_t dim, float value) {
			return std::vector<float>(dim, value);
		}

		// Helper function to generate linearly increasing vector
		std::vector<float> linearVector(size_t dim, float start = 0.0f, float step = 1.0f) {
			std::vector<float> vec(dim);
			for (size_t i = 0; i < dim; i++) {
				vec[i] = start + i * step;
			}
			return vec;
		}

	private:
		std::mt19937 gen;
		std::uniform_real_distribution<float> dist;
	};

}
