/*
 * BSD 3-Clause License
 *
 * Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CR_CPU_DATA_ALLOCATION_MANAGER_H
#define CR_CPU_DATA_ALLOCATION_MANAGER_H

#include <algorithm>

#include "array.h"
#include "type_aliases.h"

namespace crmpm
{
    struct DataBlockRange
    {
        int offset;
        int size;
    };

    class CpuDataAllocationManager
    {
    public:
        CpuDataAllocationManager(int initialCapacity = 0)
            : mCapacity(initialCapacity)
        {
            if (initialCapacity > 0)
                mFree.pushBack({0, initialCapacity});
        }

        // Request a contiguous region
        int request(int requestedSize)
        {
            if (requestedSize <= 0)
                return -1;

            int bestIndex = -1;
            int bestSize = INT_MAX;

            // Best-fit search
            for (int i = 0; i < mFree.size(); ++i)
            {
                int blockSize = mFree[i].size;
                if (blockSize >= requestedSize && blockSize < bestSize)
                {
                    bestSize = blockSize;
                    bestIndex = i;
                }
            }

            int allocOffset = 0;

            if (bestIndex == -1)
            {
                // Run out
                return -1;
            }
            else
            {
                DataBlockRange &blk = mFree[bestIndex];
                allocOffset = blk.offset;

                if (blk.size == requestedSize)
                {
                    // Perfect match, remove free block
                    mFree.remove(bestIndex);
                }
                else
                {
                    // Split: remove used portion
                    blk.offset += requestedSize;
                    blk.size -= requestedSize;
                }
            }

            return allocOffset;
        }

        // Return a block to the free list and merge if adjacent
        void release(int offset, int size)
        {
            if (size <= 0 || offset < 0 || offset + size > mCapacity)
                return;

            mFree.pushBack({offset, size});
            mergeFreeBlocks();
        }

    private:
        Array<DataBlockRange> mFree;
        size_t mCapacity = 0;

        static void sortFreeList(Array<DataBlockRange> &list)
        {
            std::sort(list.begin(), list.end(),
                      [](const DataBlockRange &a, const DataBlockRange &b)
                      { return a.offset < b.offset; });
        }

        void mergeFreeBlocks()
        {
            if (mFree.size() <= 1)
                return;

            sortFreeList(mFree);

            Array<DataBlockRange> merged;
            merged.pushBack(mFree[0]);

            for (int i = 1; i < mFree.size(); ++i)
            {
                DataBlockRange &last = merged[merged.size() - 1];
                const DataBlockRange &cur = mFree[i];

                if (last.offset + last.size == cur.offset)
                {
                    // merge adjacent
                    last.size += cur.size;
                }
                else
                {
                    merged.pushBack(cur);
                }
            }

            mFree.forceSizeUnsafe(0);
            for (int i = 0; i < merged.size(); ++i)
                mFree.pushBack(merged[i]);
        }
    };
}

#endif // !CR_CPU_DATA_ALLOCATION_MANAGER_H