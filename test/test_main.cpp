#include <catch2/catch_test_macros.hpp>	// for info on testing see https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md#test-cases-and-sections

#include "test_utils.h"

#include "hnsw/space_corr.h"

#include <vector>

using namespace testing;

static void testInstructionSets(const float ref, const std::vector<float>& vec1, const std::vector<float>& vec2, const size_t dim) {
#if defined(USE_SSE)
	float test_sse = hnswlib::CorrelationDistanceSIMD4ExtSSE(vec1.data(), vec2.data(), &dim);
	expectNear(test_sse, ref, 1e-6f, "Reference should be same as SSE results");
#endif

#if defined(USE_AVX)
	float test_avx = hnswlib::CorrelationDistanceSIMD8ExtAVX(vec1.data(), vec2.data(), &dim);
	expectNear(test_avx, ref, 1e-6f, "Reference should be same as AVX results");
#endif
}

TEST_CASE("Correlation distance reference", "[DIST][CORR]") {

	SECTION("Perfect positive correlation") {
		std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
		std::vector<float> y = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

		size_t dim = x.size();

		float ref = 0.0f;
		float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Perfect negative correlation") {
		std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
		std::vector<float> y = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

		size_t dim = x.size();

		float ref = 2.0f;
		float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Zero correlation") {
		std::vector<float> x = { 1.0f, -1.0f, 1.0f, -1.0f };
		std::vector<float> y = { 1.0f,  1.0f, -1.0f, -1.0f };

		size_t dim = x.size();

		float ref = 1.0f;
		float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Positive correlation") {
		std::vector<float> x = { 1.f, 1.5f, 3.f, 7.f };
		std::vector<float> y = { 1.f, 2.5f, 2.9f, 6.f };

		size_t dim = x.size();

		float ref = 0.024905833f;
		float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 1e-6f, "Positive correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Negative correlation") {
		std::vector<float> x = { 6.f, 6.5f, 5.9f, 3.f };
		std::vector<float> y = { 2.f, 4.5f, 5.2f, 7.f };

		size_t dim = x.size();

		float ref = 1.720908052f;
		float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 1e-6f, "Negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

}

TEST_CASE("Correlation distance edge cases", "[DIST]") {
	// Single element vectors
	std::vector<float> vec1 = { 5.0f };
	std::vector<float> vec2 = { 3.0f };
	size_t dim = 1;

	float ref = 0.0f;
	float test1 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
	expectNear(test1, ref, 1e-6f, "Single element vectors");
	testInstructionSets(ref, vec1, vec2, dim);

	// Two element vectors
	vec1 = { 1.0f, 2.0f };
	vec2 = { 3.0f, 4.0f };
	dim = 2;

	ref = 0.0f;
	float test2 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
	expectNear(test2, ref, 1e-6f, "Two element vectors");
	testInstructionSets(ref, vec1, vec2, dim);

	// Vectors with zeros
	vec1 = { 0.0f, 1.0f, 0.0f, 1.0f };
	vec2 = { 1.0f, 0.0f, 1.0f, 0.0f };
	dim = 4;

	ref = 2.0f;
	float test3 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
	expectNear(test3, ref, 1e-6f, "Vectors with zeros");
	testInstructionSets(ref, vec1, vec2, dim);
}

TEST_CASE("Correlation distance instruction sets", "[DIST][SSE][AVX]") {

	DataGenerator data;

	SECTION("Test identical vectors (should have correlation = 1, distance = 0)") {
		auto vec1 = data.linearVector(16, 1.0f, 2.0f);
		auto vec2 = vec1;
		size_t dim = vec1.size();

		float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Identical vectors should have distance ~0");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test constant vectors (undefined correlation, should return 0)") {
		auto vec1 = data.constantVector(12, 5.0f);
		auto vec2 = data.constantVector(12, 3.0f);
		size_t dim = vec1.size();

		float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Constant vectors should return distance 0");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test perfectly correlated vectors (should return 0)") {
		auto vec1 = data.linearVector(20, 0.0f, 1.0f);  // [0, 1, 2, ..., 19]
		auto vec2 = data.linearVector(20, 5.0f, 2.0f);  // [5, 7, 9, ..., 43]
		size_t dim = vec1.size();

		float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Perfect positive correlation (should return distance 0)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test perfectly anti-correlated vectors (should return 2)") {
		auto vec1 = data.linearVector(16, 0.0f, 1.0f);   // [0, 1, 2, ..., 15]
		auto vec2 = data.linearVector(16, 15.0f, -1.0f); // [15, 14, 13, ..., 0]
		size_t dim = vec1.size();

		float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(2.0f, actual, 1e-6f, "Perfect negative  correlation (should return distance 2)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test orthogonal vectors (should return 1)") {
		std::vector<float> vec1 = { 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
		std::vector<float> vec2 = { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
		size_t dim = vec1.size();

		float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(1.0f, actual, 1e-6f, "Zero correlation (should return distance 1)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Random vectors") {

		std::vector<size_t> test_sizes = { 4, 8, 12, 16, 20, 32, 64 };

		for (size_t dim : test_sizes) {
			auto vec1 = data.randomVector(dim);
			auto vec2 = data.randomVector(dim);

			float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
			testInstructionSets(actual, vec1, vec2, dim);
		}
	}


}
