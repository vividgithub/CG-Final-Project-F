#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "legacy/tf_neighbors/neighbors/neighbors.h"

using namespace tensorflow;

REGISTER_OP("FixedRadiusSearch")
	.Input("queries: float")  // 0
	.Input("supports: float") // 1
	.Input("q_batches: int32") // 2
	.Input("s_batches: int32") // 3
	.Input("radius: float") // 4
	.Input("limit: int32")  // 5
	.Output("neighbors: int32") // 0
	.Output("batches: int32") // 1
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

	// Create input shape container
	::tensorflow::shape_inference::ShapeHandle _;

	// Check inputs rank
	TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &_));  // queries: (N, 3)
	TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &_));  // supports: (N', 3)
	TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &_));  // q_batches: (B + 1)
	TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &_));  // s_batches: (B + 1)
	TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &_)); // radius: float
	TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &_));  // limit: int32

	// Create the output shape of neighbor indices
	c->set_output(0, c->UnknownShapeOfRank(1));  // neighbors: (?)

	// Create the output shape of row splits
	::tensorflow::shape_inference::DimensionHandle outputDim;  // outputDim
	TF_RETURN_IF_ERROR(c->Add(c->Dim(c->input(0), 0), 1, &outputDim)); // outputDim = N + 1
	c->set_output(1, c->Vector(outputDim));  // row splits: (N + 1, )

	return Status::OK();
});

class FixedRadiusSearchOp : public OpKernel {
public:
	explicit FixedRadiusSearchOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override
	{

		// Grab the input tensors
		const Tensor& queries_tensor = context->input(0);
		const Tensor& supports_tensor = context->input(1);
		const Tensor& q_batches_tensor = context->input(2);
		const Tensor& s_batches_tensor = context->input(3);
		const Tensor& radius_tensor = context->input(4);
		const Tensor& limit_tensor = context->input(5);

		// Get radius and limit
		float radius = radius_tensor.flat<float>().data()[0];
		int limit = limit_tensor.flat<int>().data()[0];

		// check shapes of input and weights
		const TensorShape& queries_shape = queries_tensor.shape();
		const TensorShape& supports_shape = supports_tensor.shape();
		const TensorShape& q_batches_shape = q_batches_tensor.shape();
		const TensorShape& s_batches_shape = s_batches_tensor.shape();

		// check input are [N x 3] matrices
		DCHECK_EQ(queries_shape.dims(), 2);
		DCHECK_EQ(queries_shape.dim_size(1), 3);
		DCHECK_EQ(supports_shape.dims(), 2);
		DCHECK_EQ(supports_shape.dim_size(1), 3);

		// Check that Batch lengths are vectors and same number of batch for both query and support
		DCHECK_EQ(q_batches_shape.dims(), 1);
		DCHECK_EQ(s_batches_shape.dims(), 1);
		DCHECK_EQ(q_batches_shape.dim_size(0), s_batches_shape.dim_size(0));

		// Points Dimensions
		int Nq = (int)queries_shape.dim_size(0);
		int Ns = (int)supports_shape.dim_size(0);

		// Number of row splits
		int Nb = (int)q_batches_shape.dim_size(0);

		// get the data as std vector of points
		vector<PointXYZ> queries = vector<PointXYZ>((PointXYZ*)queries_tensor.flat<float>().data(),
			(PointXYZ*)queries_tensor.flat<float>().data() + Nq);
		vector<PointXYZ> supports = vector<PointXYZ>((PointXYZ*)supports_tensor.flat<float>().data(),
			(PointXYZ*)supports_tensor.flat<float>().data() + Ns);

		// Batches lengths
		vector<int> q_batches = vector<int>((int*)q_batches_tensor.flat<int>().data(),
			(int*)q_batches_tensor.flat<int>().data() + Nb);
		vector<int> s_batches = vector<int>((int*)s_batches_tensor.flat<int>().data(),
			(int*)s_batches_tensor.flat<int>().data() + Nb);


		// Create result containers
		vector<int> indices;
		vector<int> batches;

		// Compute results
		{
			// Square radius
			float r2 = radius * radius;

			// CLoud variable
			PointCloud current_cloud;

			// Tree parameters
			nanoflann::KDTreeSingleIndexAdaptorParams tree_params(10);

			// KDTree type definition
			typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<float, PointCloud >,
				PointCloud,
				3 > my_kd_tree_t;

			// Search params
			nanoflann::SearchParams search_params;
			search_params.sorted = true;

			// Initialize output
			indices.clear();
			batches.clear();
			int nBatch = static_cast<int>(q_batches.size()) - 1;
			for (int b = 0; b < nBatch; ++b) {
				int queryStart = q_batches[b], queryEnd = q_batches[b + 1];
				int supportStart = s_batches[b], supportEnd = s_batches[b + 1];

				current_cloud.pts.clear();
				current_cloud.pts = vector<PointXYZ>(supports.begin() + supportStart, supports.begin() + supportEnd);

				// Build KD-Tree
				my_kd_tree_t index(3, current_cloud, tree_params);
				index.buildIndex();

				// Query each point in the KD-tree
				for (int i = queryStart; i < queryEnd; ++i) {
					const auto& point = queries[i];
					float pos[3] = { point.x, point.y, point.z };
					vector<pair<size_t, float>> results;

					// Query
					index.radiusSearch(pos, r2, results, search_params);

					// Push the result into indices and batches
					batches.emplace_back(results.size());
					for (const auto& result : results)
						indices.emplace_back(result.first + supportStart);  // Offset the index to the global index
				}
			}
		}

		// Create output indices
		TensorShape indices_shape({ static_cast<int64>(indices.size()) });
		Tensor* out_indices = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, indices_shape, &out_indices));
		auto out_indices_flat = out_indices->vec<int>();
		for (int i = 0; i < indices.size(); ++i)
			out_indices_flat(i) = indices[i];

		// Create batches output
		TensorShape batches_shape({ static_cast<int64>(batches.size()) + 1 });  // Output row splits
		Tensor* out_batches;
		OP_REQUIRES_OK(context, context->allocate_output(1, batches_shape, &out_batches));

		auto out_batches_flat = out_batches->vec<int>();
		out_batches_flat(0) = 0;
		for (int i = 0; i < batches.size(); ++i)
			out_batches_flat(i + 1) = out_batches_flat(i) + batches[i];
	}
};


REGISTER_KERNEL_BUILDER(Name("FixedRadiusSearch").Device(DEVICE_CPU), FixedRadiusSearchOp);