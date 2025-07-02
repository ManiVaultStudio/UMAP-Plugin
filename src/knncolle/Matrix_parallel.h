#pragma once

#include <knncolle/Matrix.hpp>

#include <memory>

/**
 * Source: https://github.com/knncolle/knncolle/blob/v3.0.1/include/knncolle/Matrix.hpp
 * MIT License, Main developer: Aaron Lun
*/
namespace knncolle {

    /**
     * @brief Extractor for a `ParallelMatrix`.
     *
     * This should be typically constructed by calling `ParallelMatrix::new_extractor()`.
     *
     * @tparam Data_ Numeric type of the data.
     */
    template<typename Data_>
    class ParallelMatrixExtractor final : public MatrixExtractor<Data_> {
    public:
        /**
         * @cond
         */
        ParallelMatrixExtractor(const Data_* data, std::size_t dim) : my_data(data), my_dim(dim) {}
        /**
         * @endcond
         */

    private:
        const Data_* my_data;
        std::size_t my_dim;
        std::size_t at = 0;

    public:
        const Data_* next() override {
            return my_data + (at++) * my_dim; // already std::size_t's to avoid overflow during multiplication.
        }

        const Data_* get(std::size_t point) const {
            return my_data + point * my_dim; // already std::size_t's to avoid overflow during multiplication.
        }
    };

    /**
     * @brief Simple wrapper for an in-memory matrix.
     *
     * This defines a simple column-major matrix of observations where the columns are observations and the rows are dimensions.
     * It is compatible with the compile-time interface described in `MockMatrix`.
     *
     * @tparam Index_ Integer type of the observation indices.
     * @tparam Data_ Numeric type of the data.
     */
    template<typename Index_, typename Data_>
    class ParallelMatrix final : public Matrix<Index_, Data_> {
    public:
        /**
         * @param num_dimensions Number of dimensions.
         * @param num_observations Number of observations.
         * @param[in] data Pointer to an array of length `num_dim * num_obs`, containing a column-major matrix of observation data.
         * It is expected that the array will not be deallocated during the lifetime of this `SimpleMatrix` instance.
         */
        ParallelMatrix(std::size_t num_dimensions, Index_ num_observations, const Data_* data) :
        my_num_dim(num_dimensions), my_num_obs(num_observations), my_data(data) {}

    private:
        std::size_t my_num_dim;
        Index_ my_num_obs;
        const Data_* my_data;

    public:
        Index_ num_observations() const override {
            return my_num_obs;
        }

        std::size_t num_dimensions() const override {
            return my_num_dim;
        }

        std::unique_ptr<MatrixExtractor<Data_> > new_extractor() const override {
            return std::make_unique<ParallelMatrixExtractor<Data_> >(my_data, my_num_dim);
        }
    };

}
