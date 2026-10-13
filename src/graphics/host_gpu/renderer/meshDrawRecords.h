#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWRECORDS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWRECORDS_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

// Parameter records of mesh-emulated draws. Mesh shaders read the record whose device address is
// in push-constant dwords 0 and 1. Direct draws write it on the CPU; draws from GPU-written
// indirect arguments have a compute pass build it, with the mesh dispatch, on the GPU.
class MeshDrawRecords {
public:
	static constexpr uint32_t RecordDwords = 6;

	// Mesh layout of an indexed draw that a GPU-args draw needs to size its dispatch.
	struct IndirectParams {
		uint32_t index_limit          = 0; // INDEX_BUFFER_SIZE in elements.
		uint32_t element_size         = 0;
		uint64_t index_base           = 0;
		uint32_t primitive_size       = 0;
		uint32_t primitive_step       = 0;
		uint32_t primitives_per_group = 0;
	};

	struct IndirectDraw {
		vk::DeviceAddress record         = 0;
		vk::Buffer        command        = nullptr;
		uint64_t          command_offset = 0;
	};

	MeshDrawRecords(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshDrawRecords();
	KYTY_CLASS_NO_COPY(MeshDrawRecords);

	[[nodiscard]] vk::DeviceAddress Write(const std::array<uint32_t, RecordDwords>& record);
	// Records the compute pass that turns DrawIndexedIndirect arguments at `args_offset` in
	// `args` into a record and a mesh dispatch. Must be recorded outside a rendering instance.
	[[nodiscard]] IndirectDraw RecordIndirect(vk::CommandBuffer command, vk::Buffer args,
	                                          uint64_t args_offset, uint64_t args_size,
	                                          const IndirectParams& params);

private:
	GraphicContext&         m_graphics;
	StreamBuffer            m_ring;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_pipeline          = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MESHDRAWRECORDS_H_
