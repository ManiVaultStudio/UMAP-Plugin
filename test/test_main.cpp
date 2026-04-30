#include <knncolle/Builder.hpp>
#include <knncolle_hnsw/knncolle_hnsw.hpp>
#include <knncolle_hnsw/distances.hpp>

#include <catch2/catch_timer.hpp>

#include "util/hnsw_space_corr.h"
#include "util/knncolle_matrix_parallel.h"
#include "util/knncolle_hnsw_parallel.h"
#include "util/knncolle_find_nearest_neighbors.h"

#include "test_utils.h"

#include <cstdint>
#include <format>
#include <thread>
#include <vector>

using namespace testing;

/// /////////// ///
/// Definitions ///
/// //////////  ///

using integer_type  = int32_t;
using scalar_type   = float;
using DataMatrix	= knncolle::SimpleMatrix< /* observation index */ integer_type, /* data type */ scalar_type>;
using DataMatrixPar	= knncolle::ParallelMatrix< /* observation index */ integer_type, /* data type */ scalar_type>;
using KnnHnsw       = knncolle_hnsw::HnswBuilder<integer_type, scalar_type, scalar_type, DataMatrix>;
using KnnHnswPar    = knncolle_hnsw::HnswBuilderParallel<integer_type, scalar_type, scalar_type, DataMatrixPar>;
using KnnList       = knncolle::NeighborList<integer_type, scalar_type>;

/// ////////// ///
///   Helper   ///
/// ////////// ///

namespace
{
	void testInstructionSets(const float ref, const std::vector<float>& vec1, const std::vector<float>& vec2, const size_t dim) {
#ifdef USE_SSE
		const float test_sse = hnswlib::CorrelationDistanceSIMD4ExtSSE(vec1.data(), vec2.data(), &dim);
		expectNear(test_sse, ref, 1e-6f, "Reference should be same as SSE results");
#endif

#ifdef USE_AVX
		const float test_avx = hnswlib::CorrelationDistanceSIMD8ExtAVX(vec1.data(), vec2.data(), &dim);
		expectNear(test_avx, ref, 1e-6f, "Reference should be same as AVX results");
#endif
	}
}

TEST_CASE("Correlation distance reference", "[DIST][CORR]") {

	SECTION("Perfect positive correlation") {
		info("TEST: Correlation distance reference -> Perfect positive correlation");
		const std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
		const std::vector<float> y = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

		const size_t dim = x.size();

		const float ref = 0.0f;
		const float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Perfect negative correlation") {
		info("TEST: Correlation distance reference -> Perfect negative correlation");
		const std::vector<float> x = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
		const std::vector<float> y = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

		const size_t dim = x.size();

		const float ref = 2.0f;
		const float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Zero correlation") {
		info("TEST: Correlation distance reference -> Zero correlation");
		const std::vector<float> x = { 1.0f, -1.0f, 1.0f, -1.0f };
		const std::vector<float> y = { 1.0f,  1.0f, -1.0f, -1.0f };

		const size_t dim = x.size();

		const float ref = 1.0f;
		const float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 0, "Perfect negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Positive correlation") {
		info("TEST: Correlation distance reference -> Positive correlation");
		const std::vector<float> x = { 1.f, 1.5f, 3.f, 7.f };
		const std::vector<float> y = { 1.f, 2.5f, 2.9f, 6.f };

		const size_t dim = x.size();

		const float ref = 0.024905833f;
		const float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 1e-6f, "Positive correlation");

		testInstructionSets(ref, x, y, dim);
	}

	SECTION("Negative correlation") {
		info("TEST: Correlation distance reference -> Negative correlation");
		const std::vector<float> x = { 6.f, 6.5f, 5.9f, 3.f };
		const std::vector<float> y = { 2.f, 4.5f, 5.2f, 7.f };

		const size_t dim = x.size();

		const float ref = 1.720908052f;
		const float test = hnswlib::CorrelationDistance(x.data(), y.data(), &dim);

		expectNear(test, ref, 1e-6f, "Negative correlation");

		testInstructionSets(ref, x, y, dim);
	}

}

TEST_CASE("Correlation distance edge cases", "[DIST]") {
	info("TEST: Correlation distance edge cases");

	// Single element vectors
	{
		const std::vector<float> vec1 = { 5.0f };
		const std::vector<float> vec2 = { 3.0f };
		const size_t dim = 1;

		const float ref = 0.0f;
		const float test1 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
		expectNear(test1, ref, 1e-6f, "Single element vectors");
		testInstructionSets(ref, vec1, vec2, dim);
	}

	// Two element vectors
	{
		const std::vector<float> vec1 = { 1.0f, 2.0f };
		const std::vector<float> vec2 = { 3.0f, 4.0f };
		const size_t dim = 2;

		const float ref = 0.0f;
		const float test2 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
		expectNear(test2, ref, 1e-6f, "Two element vectors");
		testInstructionSets(ref, vec1, vec2, dim);
	}

	// Vectors with zeros
	{
		const std::vector<float> vec1 = { 0.0f, 1.0f, 0.0f, 1.0f };
		const std::vector<float> vec2 = { 1.0f, 0.0f, 1.0f, 0.0f };
		const size_t dim = 4;

		const float ref = 2.0f;
		const float test3 = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
		expectNear(test3, ref, 1e-6f, "Vectors with zeros");
		testInstructionSets(ref, vec1, vec2, dim);
	}
}

TEST_CASE("Correlation distance instruction sets", "[DIST][SSE][AVX]") {

	DataGenerator data;

	SECTION("Test identical vectors (should have correlation = 1, distance = 0)") {
		info("TEST: Correlation distance instruction sets -> Test identical vectors");

		const auto vec1 = data.linearVector(16, 1.0f, 2.0f);
		const auto vec2 = vec1;
		const size_t dim = vec1.size();

		const float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Identical vectors should have distance ~0");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test constant vectors (undefined correlation, should return 0)") {
		info("TEST: Correlation distance instruction sets -> Test constant vectors");

		const auto vec1 = data.constantVector(12, 5.0f);
		const auto vec2 = data.constantVector(12, 3.0f);
		const size_t dim = vec1.size();

		const float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Constant vectors should return distance 0");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test perfectly correlated vectors (should return 0)") {
		info("TEST: Correlation distance instruction sets -> Test perfectly correlated vectors");

		const auto vec1 = data.linearVector(20, 0.0f, 1.0f);  // [0, 1, 2, ..., 19]
		const auto vec2 = data.linearVector(20, 5.0f, 2.0f);  // [5, 7, 9, ..., 43]
		const size_t dim = vec1.size();

		const float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(0.0f, actual, 1e-6f, "Perfect positive correlation (should return distance 0)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test perfectly anti-correlated vectors (should return 2)") {
		info("TEST: Correlation distance instruction sets -> Test perfectly anti-correlated vectors");

		const auto vec1 = data.linearVector(16, 0.0f, 1.0f);   // [0, 1, 2, ..., 15]
		const auto vec2 = data.linearVector(16, 15.0f, -1.0f); // [15, 14, 13, ..., 0]
		const size_t dim = vec1.size();

		const float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(2.0f, actual, 1e-6f, "Perfect negative  correlation (should return distance 2)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Test orthogonal vectors (should return 1)") {
		info("TEST: Correlation distance instruction sets -> Test orthogonal vectors");

		const std::vector<float> vec1 = { 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
		const std::vector<float> vec2 = { 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f };
		const size_t dim = vec1.size();

		const float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);

		expectNear(1.0f, actual, 1e-6f, "Zero correlation (should return distance 1)");
		testInstructionSets(actual, vec1, vec2, dim);
	}

	SECTION("Random vectors") {
		info("TEST: Correlation distance instruction sets -> Random vectors");

		const std::vector<size_t> test_sizes = { 4, 8, 12, 16, 20, 32, 64 };

		for (size_t dim : test_sizes) {
			auto vec1 = data.randomVector(dim);
			auto vec2 = data.randomVector(dim);

			float actual = hnswlib::CorrelationDistance(vec1.data(), vec2.data(), &dim);
			testInstructionSets(actual, vec1, vec2, dim);
		}
	}

}

TEST_CASE("Parallel HNSW", "[DIST][KNN]") {
	info("TEST: Parallel HNSW");

	DataGenerator<scalar_type> gen;

	constexpr size_t numPoints = 10'000;
	constexpr size_t numDim = 16;

	const std::vector<scalar_type> data = gen.randomMatrix(numDim, numPoints);

	knncolle_hnsw::HnswOptions opt;
	opt.num_links = 16;
	opt.ef_search = 250;
	opt.ef_construction = 250;

	const auto mat = DataMatrix(numDim, numPoints, data.data());
	const auto matPar = DataMatrixPar(numDim, numPoints, data.data());

	const int numThreads = static_cast<int>(std::thread::hardware_concurrency());

	Catch::Timer timer;

	SECTION("Exact equality for low number of neighbors") {
		info("TEST: Parallel HNSW -> Exact equality for low number of neighbors");

		constexpr size_t numNeighbors = 25;

		KnnList nn_seq;
		KnnList nn_parQue;
		KnnList nn_parAll;

		{
			timer.start();
			info("Search sequential");
			auto searcherSeq = KnnHnsw(knncolle_hnsw::configure_euclidean_distance<scalar_type>(), opt).build_unique(mat);
			nn_seq = knncolle::find_nearest_neighbors_custom<integer_type, scalar_type, scalar_type>(*searcherSeq, numNeighbors, 1);
			printDuration(timer.getElapsedMicroseconds());
		}

		{
			timer.start();
			info("Search parallel: query");
			auto searcherParQue = KnnHnsw(knncolle_hnsw::configure_euclidean_distance<scalar_type>(), opt).build_unique(mat);
			nn_parQue = knncolle::find_nearest_neighbors_custom<integer_type, scalar_type, scalar_type>(*searcherParQue, numNeighbors, numThreads);
			printDuration(timer.getElapsedMicroseconds());
		}

		{
			timer.start();
			info("Search parallel: addition and query");
			auto searcherParAll = KnnHnswPar(knncolle_hnsw::configure_euclidean_distance<scalar_type>(), opt).build_unique(matPar);
			nn_parAll = knncolle::find_nearest_neighbors_custom<integer_type, scalar_type, scalar_type>(*searcherParAll, numNeighbors, numThreads);
			printDuration(timer.getElapsedMicroseconds());
		}

		info("Check equality");

		REQUIRE(nn_seq.size() == numPoints);
		REQUIRE(nn_parQue.size() == numPoints);
		REQUIRE(nn_parAll.size() == numPoints);

		for (size_t i = 0; i < numPoints; i++) {
			REQUIRE(nn_seq[i].size() == numNeighbors);
			REQUIRE(nn_parQue[i].size() == numNeighbors);
			REQUIRE(nn_parAll[i].size() == numNeighbors);

			for (size_t j = 0; j < numNeighbors; j++) {

				if (nn_seq[i][j].first != nn_parQue[i][j].first) {
					print(nn_seq[i]);
					print(nn_parQue[i]);
				}

				if (!nearlyEqual(nn_seq[i][j].second, nn_parQue[i][j].second)) {
					print(nn_seq[i]);
					print(nn_parQue[i]);
				}

				if (nn_seq[i][j].first != nn_parAll[i][j].first) {
					print(nn_seq[i]);
					print(nn_parAll[i]);
				}

				if (!nearlyEqual(nn_seq[i][j].second, nn_parAll[i][j].second)) {
					print(nn_seq[i]);
					print(nn_parAll[i]);
				}

				REQUIRE(nn_seq[i][j].first == nn_parQue[i][j].first);
				REQUIRE(nearlyEqual(nn_seq[i][j].second, nn_parQue[i][j].second));

				REQUIRE(nn_seq[i][j].first == nn_parAll[i][j].first);
				REQUIRE(nearlyEqual(nn_seq[i][j].second, nn_parAll[i][j].second));
			}
		}
	}


	SECTION("Recall for larger number of neighbors") {
		info("TEST: Parallel HNSW -> Recall for larger number of neighbors");

		constexpr size_t numNeighbors = 100;

		KnnList nn_seq;
		KnnList nn_parAll;

		{
			timer.start();
			info("Search sequential");
			auto searcherSeq = KnnHnsw(knncolle_hnsw::configure_euclidean_distance<scalar_type>(), opt).build_unique(mat);
			nn_seq = knncolle::find_nearest_neighbors_custom<integer_type, scalar_type, scalar_type>(*searcherSeq, numNeighbors, 1);
			printDuration(timer.getElapsedMicroseconds());
		}

		{
			timer.start();
			info("Search parallel: addition and query");
			auto searcherParAll = KnnHnswPar(knncolle_hnsw::configure_euclidean_distance<scalar_type>(), opt).build_unique(matPar);
			nn_parAll = knncolle::find_nearest_neighbors_custom<integer_type, scalar_type, scalar_type>(*searcherParAll, numNeighbors, numThreads);
			printDuration(timer.getElapsedMicroseconds());
		}

		float correct = 0;

		REQUIRE(nn_seq.size() == numPoints);
		REQUIRE(nn_parAll.size() == numPoints);

		for (size_t i = 0; i < numPoints; i++) {
			REQUIRE(nn_seq[i].size() == numNeighbors);
			REQUIRE(nn_parAll[i].size() == numNeighbors);

			for (size_t j = 0; j < numNeighbors; j++) {
				if (nn_seq[i][j].first == nn_parAll[i][j].first &&
					std::abs(nn_seq[i][j].second - nn_parAll[i][j].second) < 1e-6f) {
					correct++;
				}
			}
		}

		const float recall = correct / (numNeighbors * numPoints);
		info(std::format("Recall: {}", recall));
	}
}