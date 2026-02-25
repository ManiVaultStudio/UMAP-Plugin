#pragma once

#include <knncolle/Prebuilt.hpp>

#include <vector>
#include <utility>
#include <type_traits>

#include "sanisizer/sanisizer.hpp"
#ifndef KNNCOLLE_CUSTOM_PARALLEL
#include "subpar/subpar.hpp"
#endif

namespace knncolle
{
    /** From <knncolle/find_nearest_neighbors.hpp>
    * MSVC v143 seems to have some issues here
    */
    template<typename Index_, typename Data_, typename Distance_>
    NeighborList<Index_, Distance_> find_nearest_neighbors_custom(const Prebuilt<Index_, Data_, Distance_>& index, int k, int num_threads = 1) {
        const Index_ nobs = index.num_observations();
        k = cap_k(k, nobs);
        auto output = sanisizer::create<NeighborList<Index_, Distance_> >(sanisizer::attest_gez(nobs));

        parallelize(num_threads, nobs, [&](int, Index_ start, Index_ length) -> void {
            auto sptr = index.initialize_known();
            std::vector<Index_> indices;
            std::vector<Distance_> distances;
            for (Index_ i = start, end = start + length; i < end; ++i) {
                sptr->search(i, k, &indices, &distances);
                const auto actual_k = indices.size();
                output[i].reserve(actual_k);
                for (std::size_t j = 0; j < actual_k; ++j) {
                    output[i].emplace_back(indices[j], distances[j]);
                }
            }
            });

        return output;
    }

}
