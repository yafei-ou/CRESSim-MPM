/**
 * Implementation referred to PhysX's GuAABBTree.h, GuAABBTreeNode.h,
 * GuAABBTreeBounds.h, GuAABBTreeBuildStats.h, GuAABBTree.cpp, GuAABBTreeQuery.h,
 * and GuDistancePointTriangle.cpp
 *
 * https://github.com/NVIDIA-Omniverse/PhysX/tree/dd587fedd79836442a4117164ea8c46685453c34/physx/source/geomutils/src
 *
 * Original code licensed under the BSD 3-Clause License by NVIDIA Corporation.
 * 
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *  * Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *  contributors may be used to endorse or promote products derived
 *  from this software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
 *  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 *  Copyright (c) 2008-2024 NVIDIA Corporation. All rights reserved.
 *  Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
 *  Copyright (c) 2001-2004 NovodeX AG. All rights reserved.
 *
 * Modifications by:
 *  Copyright (c) 2025, Yafei Ou and Mahdi Tavakoli
 *  Licensed under the BSD 3-Clause License.
 */

#ifndef CR_AABB_TREE_H
#define CR_AABB_TREE_H

#include "constants.h"
#include <cassert>
#include <utility>
#include <algorithm>

#include "type_aliases.h"
#include "preprocessor.h"
#include "bounds3.h"
#include "array.h"
#include "math_helpers.h"
#include "engine_export.h"

namespace crmpm
{
    struct AABBTreeNode
    {
    public:
        CR_FORCE_INLINE AABBTreeNode() {}
        CR_FORCE_INLINE ~AABBTreeNode() {}

        CR_FORCE_INLINE unsigned int isLeaf() const { return mData & 1; }
        CR_FORCE_INLINE const unsigned int *getPrimitives(const unsigned int *base) const { return base + (mData >> 5); }
        CR_FORCE_INLINE unsigned int *getPrimitives(unsigned int *base) { return base + (mData >> 5); }
        CR_FORCE_INLINE unsigned int getPrimitiveIndex() const { return mData >> 5; }
        CR_FORCE_INLINE unsigned int getNbPrimitives() const { return (mData >> 1) & 15; }
        CR_FORCE_INLINE unsigned int getPosIndex() const { return mData >> 1; }
        CR_FORCE_INLINE unsigned int getNegIndex() const { return (mData >> 1) + 1; }
        CR_FORCE_INLINE const AABBTreeNode *getPos(const AABBTreeNode *base) const { return base + (mData >> 1); }
        CR_FORCE_INLINE const AABBTreeNode *getNeg(const AABBTreeNode *base) const
        {
            const AABBTreeNode *P = getPos(base);
            return P ? P + 1 : NULL;
        }
        CR_FORCE_INLINE AABBTreeNode *getPos(AABBTreeNode *base) { return base + (mData >> 1); }
        CR_FORCE_INLINE AABBTreeNode *getNeg(AABBTreeNode *base)
        {
            AABBTreeNode *P = getPos(base);
            return P ? P + 1 : NULL;
        }

        Bounds3 mBV;        // Global bounding-volume enclosing all the node-related primitives
        unsigned int mData; // 27 bits node or prim index|4 bits #prims|1 bit leaf
    };

    class AABBTreeBuildParams
    {
    public:
        AABBTreeBuildParams(unsigned int limit = 1, unsigned int nb_prims = 0, const Bounds3 *bounds = NULL) : mLimit(limit), mNbPrimitives(nb_prims), mBounds(bounds), mCache(NULL) {}

        ~AABBTreeBuildParams()
        {
            reset();
        }

        CR_FORCE_INLINE void reset()
        {
            mLimit = mNbPrimitives = 0;
            mBounds = NULL;
            delete[] mCache;
        }

        unsigned int mLimit;        //!< Limit number of primitives / node. If limit is 1, build a complete tree (2*N-1 nodes)
        unsigned int mNbPrimitives; //!< Number of (source) primitives.
        const Bounds3 *mBounds;     //!< Shortcut to an app-controlled array of AABBs.
        mutable Vec3f3 *mCache;      //!< Cache for AABB centers - managed by build code.
    };

    struct BuildStats
    {
        BuildStats() : mCount(0), mTotalPrims(0) {}

        unsigned int mCount;
        unsigned int mTotalPrims;
    };

    class NodeAllocator;

    class CRMPM_ENGINE_API AABBTreeBuildNode
    {
    public:
        CR_FORCE_INLINE AABBTreeBuildNode() : mBV(), mPos(0), mNodeIndex(0), mNbPrimitives(0) {}
        CR_FORCE_INLINE ~AABBTreeBuildNode() {}

        CR_FORCE_INLINE const Bounds3 &getAABB() const { return mBV; }
        CR_FORCE_INLINE const AABBTreeBuildNode *getPos() const { return mPos; }
        CR_FORCE_INLINE const AABBTreeBuildNode *getNeg() const
        {
            const AABBTreeBuildNode *P = mPos;
            return P ? P + 1 : NULL;
        }

        CR_FORCE_INLINE bool isLeaf() const { return !getPos(); }

        Bounds3 mBV;                   //!< Global bounding-volume enclosing all the node-related primitives
        const AABBTreeBuildNode *mPos; //!< "Positive" & "Negative" children

        unsigned int mNodeIndex;    //!< Index of node-related primitives (in the tree's mIndices array)
        unsigned int mNbPrimitives; //!< Number of primitives for this node

        CR_FORCE_INLINE unsigned int getNbPrimitives() const { return mNbPrimitives; }
        CR_FORCE_INLINE const unsigned int *getPrimitives(const unsigned int *base) const { return base + mNodeIndex; }
        CR_FORCE_INLINE unsigned int *getPrimitives(unsigned int *base) { return base + mNodeIndex; }

        void subdivide(const AABBTreeBuildParams &params, BuildStats &stats, NodeAllocator &allocator, unsigned int *const indices);
    };

    class NodeAllocator
    {
    public:
        NodeAllocator() : mPool(NULL), mCurrentSlabIndex(0), mTotalNbNodes(0) {}
        ~NodeAllocator()
        {
            const unsigned int nbSlabs = mSlabs.size();
            for (unsigned int i = 0; i < nbSlabs; i++)
            {
                Slab &s = mSlabs[i];
                delete[] s.mPool;
            }

            mCurrentSlabIndex = 0;
            mTotalNbNodes = 0;
            mSlabs.clear();
        }

        void init(unsigned int nbPrimitives, unsigned int limit)
        {
            // Just allocate enough for a complete tree
            unsigned int maxNodes = 2 * nbPrimitives - 1;
            const unsigned int estimatedFinalSize = maxNodes <= 1024 ? maxNodes : maxNodes / limit;
            mPool = new AABBTreeBuildNode[estimatedFinalSize];

            mPool->mNodeIndex = 0;
            mPool->mNbPrimitives = nbPrimitives;
            mSlabs.pushBack(Slab(mPool, 1, maxNodes));

            mCurrentSlabIndex = 0;
            mTotalNbNodes = 1;
        }

        /**
         * @brief Returns a pointer to 2 new build nodes for splitting
         */
        AABBTreeBuildNode *getBiNode()
        {
            mTotalNbNodes += 2;
            Slab &currentSlab = mSlabs[mCurrentSlabIndex];
            if (currentSlab.mNbUsedNodes + 2 <= currentSlab.mMaxNbNodes)
            {
                AABBTreeBuildNode *biNode = currentSlab.mPool + currentSlab.mNbUsedNodes;
                currentSlab.mNbUsedNodes += 2;
                return biNode;
            }
            else
            {
                // Allocate new slab
                const unsigned int size = 1024;
                AABBTreeBuildNode *pool = new AABBTreeBuildNode[size];

                mSlabs.pushBack(Slab(pool, 2, size));
                mCurrentSlabIndex++;
                return pool;
            }
        }

        struct Slab
        {
            Slab() : mPool(NULL), mNbUsedNodes(0), mMaxNbNodes(0) {}
            Slab(AABBTreeBuildNode *pool, unsigned int used, unsigned int maxN)
                : mPool(pool), mNbUsedNodes(used), mMaxNbNodes(maxN) {}
            AABBTreeBuildNode *mPool;
            unsigned int mNbUsedNodes;
            unsigned int mMaxNbNodes;
        };

        Array<Slab> mSlabs;
        AABBTreeBuildNode *mPool;
        unsigned int mCurrentSlabIndex;
        unsigned int mTotalNbNodes;
    };

    CRMPM_ENGINE_API unsigned int *buildAABBTree(const AABBTreeBuildParams &params, NodeAllocator &nodeAllocator, BuildStats &stats);

    CRMPM_ENGINE_API void flattenTree(const NodeAllocator &nodeAllocator, AABBTreeNode *dest, const unsigned int *remap);

    /**
     * Build AABB tree from nbBounds bounds and write to tree.
     */
    CRMPM_ENGINE_API void buildAABBTree(unsigned int nbBounds, const Bounds3 *CR_RESTRICT bounds, Array<AABBTreeNode> &tree);

    // AABB tree query

    enum class TraversalControl
    {
        eDontGoDeeper,
        eGoDeeper,
        eGoDeeperNegFirst,
        eAbort
    };

    template <typename T>
    void traverseBVH(const AABBTreeNode *nodes, T &traversalController, int rootNodeIndex = 0)
    {
        int index = rootNodeIndex;

        Array<int> todoStack;

        while (true)
        {
            const AABBTreeNode &a = nodes[index];

            TraversalControl control = traversalController.analyze(a, index);
            if (control == TraversalControl::eAbort)
                return;
            if (!a.isLeaf() && (control == TraversalControl::eGoDeeper || control == TraversalControl::eGoDeeperNegFirst))
            {
                if (control == TraversalControl::eGoDeeperNegFirst)
                {
                    todoStack.pushBack(a.getPosIndex());
                    index = a.getNegIndex(); // index gets processed next - assign negative index to it
                }
                else
                {
                    todoStack.pushBack(a.getNegIndex());
                    index = a.getPosIndex(); // index gets processed next - assign positive index to it
                }
                continue;
            }
            if (todoStack.empty())
                break;
            index = todoStack.popBack();
        }
    }

    CRMPM_ENGINE_API float distancePointTriangleSquared2UnitBox(const Vec3f3 queryPoint, const Vec3f3 triA, const Vec3f3 triB, const Vec3f3 triC, float &u, float &v, Vec3f3 &closestP);

    class CRMPM_ENGINE_API ClosestDistanceToTrimeshTraversalController
    {
    private:
        float mClosestDistanceSquared;
        const unsigned int *mTriangles;
        const Vec3f3 *mPoints;
        const AABBTreeNode *mNodes;
        Vec3f3 mQueryPoint;
        Vec3f3 mClosestPoint;
        int mClosestTriId;

    public:
        CR_FORCE_INLINE ClosestDistanceToTrimeshTraversalController() {}

        CR_FORCE_INLINE ClosestDistanceToTrimeshTraversalController(const unsigned int *triangles, const Vec3f3 *points, AABBTreeNode *nodes) : mClosestDistanceSquared(CR_MAX_F32), mTriangles(triangles), mPoints(points), mNodes(nodes), mQueryPoint(0.0f), mClosestPoint(0.0f), mClosestTriId(-1)
        {
            initialize(triangles, points, nodes);
        }

        void initialize(const unsigned int *triangles, const Vec3f3 *points, AABBTreeNode *nodes)
        {
            mTriangles = triangles;
            mPoints = points;
            mNodes = nodes;
            mQueryPoint = Vec3f3(0.0f);
            mClosestPoint = Vec3f3(0.0f);
            mClosestTriId = -1;
            mClosestDistanceSquared = CR_MAX_F32;
        }

        CR_FORCE_INLINE void setQueryPoint(const Vec3f3 &queryPoint)
        {
            mQueryPoint = queryPoint;
            mClosestDistanceSquared = CR_MAX_F32;
            mClosestPoint = Vec3f3(0.0f);
            mClosestTriId = -1;
        }

        CR_FORCE_INLINE const Vec3f3 &getClosestPoint() const
        {
            return mClosestPoint;
        }

        CR_FORCE_INLINE float distancePointBoxSquared(const Bounds3 &box, const Vec3f3 &point)
        {
            Vec3f3 closestPt = max(box.minimum.cast<float, 3>(), min(box.maximum.cast<float, 3>(), point));

            return (closestPt - point).squaredNorm();
        }

        TraversalControl analyze(const AABBTreeNode &node, int);

        CR_FORCE_INLINE int getClosestTriId() const { return mClosestTriId; }

        CR_FORCE_INLINE void setClosestStart(const float closestDistanceSquared, int closestTriangle, const Vec3f3 &closestPoint)
        {
            mClosestDistanceSquared = closestDistanceSquared;
            mClosestTriId = closestTriangle;
            mClosestPoint = closestPoint;
        }
    };

    struct EdgeHash
    {
        std::size_t operator()(const std::pair<int, int> &edge) const
        {
            return std::hash<int>()(edge.first) ^ std::hash<int>()(edge.second);
        }
    };

    class CRMPM_ENGINE_API ClosestDistanceProjectionToTrimeshTraversalController
    {
    private:
        float mClosestDistanceSquared;
        const unsigned int *mTriangles;
        const Vec3f3 *mPoints;
        const AABBTreeNode *mNodes;
        Vec3f3 mQueryPoint;
        Vec3f3 mClosestPoint;
        int mClosestTriId;
        bool mProjectionOnSurface;
        int mSide;
        HashMap<Pair<int, int>, Array<int>, EdgeHash> *mEdgeTriangleMap;
        Array<int> *mEdgeVertices;

    public:
        CR_FORCE_INLINE ClosestDistanceProjectionToTrimeshTraversalController() {}

        CR_FORCE_INLINE ClosestDistanceProjectionToTrimeshTraversalController(
            const unsigned int *triangles,
            const Vec3f3 *points,
            AABBTreeNode *nodes,
            HashMap<Pair<int, int>, Array<int>, EdgeHash> *edgeTriangleMap,
            Array<int> *edgeVertices) : mClosestDistanceSquared(CR_MAX_F32), mTriangles(triangles), mPoints(points), mNodes(nodes), mQueryPoint(0.0f), mClosestPoint(0.0f), mClosestTriId(-1), mEdgeTriangleMap(edgeTriangleMap), mEdgeVertices(edgeVertices)
        {
            initialize(triangles, points, nodes);
        }

        CR_FORCE_INLINE void initialize(const unsigned int *triangles, const Vec3f3 *points, AABBTreeNode *nodes)
        {
            mTriangles = triangles;
            mPoints = points;
            mNodes = nodes;
            mQueryPoint = Vec3f3(0.0f);
            mClosestPoint = Vec3f3(0.0f);
            mClosestTriId = -1;
            mClosestDistanceSquared = CR_MAX_F32;
            mProjectionOnSurface = true;
            mSide = 0;
        }

        CR_FORCE_INLINE void setQueryPoint(const Vec3f3 &queryPoint)
        {
            mQueryPoint = queryPoint;
            mClosestDistanceSquared = CR_MAX_F32;
            mClosestPoint = Vec3f3(0.0f);
            mClosestTriId = -1;
            mSide = 0;
            mProjectionOnSurface = true;
        }

        CR_FORCE_INLINE const Vec3f3 &getClosestPoint() const
        {
            return mClosestPoint;
        }

        CR_FORCE_INLINE const bool &getProjectionOnSurface() const
        {
            return mProjectionOnSurface;
        }

        CR_FORCE_INLINE const int &getSide() const
        {
            return mSide;
        }

        CR_FORCE_INLINE float distancePointBoxSquared(const Bounds3 &box, const Vec3f3 &point)
        {
            Vec3f3 closestPt = max(box.minimum.cast<float, 3>(), min(box.maximum.cast<float, 3>(), point));

            return (closestPt - point).squaredNorm();
        }

        TraversalControl analyze(const AABBTreeNode &node, int);

        CR_FORCE_INLINE int getClosestTriId() const { return mClosestTriId; }

        CR_FORCE_INLINE void setClosestStart(const float closestDistanceSquared, int closestTriangle, const Vec3f3 &closestPoint)
        {
            mClosestDistanceSquared = closestDistanceSquared;
            mClosestTriId = closestTriangle;
            mClosestPoint = closestPoint;
        }
    };

} // namespace crmpm

#endif // !CR_AABB_TREE_H