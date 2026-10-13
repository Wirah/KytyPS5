#include "graphics/host_gpu/renderer/meshDrawRecords.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_tiler_shaders/mesh_draw_indirect_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint64_t RingSize      = 4ull * 1024 * 1024;
constexpr uint64_t RecordBytes   = 32;
constexpr uint64_t CommandBytes  = sizeof(VkDrawMeshTasksIndirectCommandEXT);
constexpr uint32_t CommandWord   = RecordBytes / sizeof(uint32_t);
constexpr uint64_t IndirectBytes = RecordBytes + 16;

struct IndirectPushData {
	uint32_t args_word;
	uint32_t draw_word;
	uint32_t command_word;
	uint32_t index_limit;
	uint32_t element_size;
	uint32_t index_base_low;
	uint32_t index_base_high;
	uint32_t primitive_size;
	uint32_t primitive_step;
	uint32_t primitives_per_group;
	uint32_t max_groups_x;
	uint32_t max_groups_y;
	uint32_t max_groups_total;
};

static_assert(MeshDrawRecords::RecordDwords * sizeof(uint32_t) <= RecordBytes);
static_assert(RecordBytes + CommandBytes <= IndirectBytes);

} // namespace

MeshDrawRecords::MeshDrawRecords(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics),
      m_ring(graphics, scheduler, MemoryUsage::Stream, RingSize,
             vk::BufferUsageFlagBits::eShaderDeviceAddress) {
	SetVulkanObjectNameF(graphics.device, m_ring.Handle(), "Mesh Draw Records");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	    "create mesh-indirect descriptor layout");

	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(IndirectPushData)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_descriptor_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &m_pipeline_layout),
	                     "create mesh-indirect pipeline layout");

	const auto module = CompileSPV(MESH_DRAW_INDIRECT_SPV, graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result =
	    graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-indirect pipeline");
	SetVulkanObjectNameF(graphics.device, m_pipeline, "Mesh Draw Indirect");
}

MeshDrawRecords::~MeshDrawRecords() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
}

vk::DeviceAddress MeshDrawRecords::Write(const std::array<uint32_t, RecordDwords>& record) {
	const auto [mapped, offset] = m_ring.Map(RecordBytes, 16);
	EXIT_IF(mapped == nullptr);
	std::memcpy(mapped, record.data(), sizeof(record));
	m_ring.Commit();
	return m_ring.BufferDeviceAddress() + offset;
}

MeshDrawRecords::IndirectDraw MeshDrawRecords::RecordIndirect(vk::CommandBuffer command,
                                                              vk::Buffer args, uint64_t args_offset,
                                                              uint64_t              args_size,
                                                              const IndirectParams& params) {
	EXIT_IF(args == nullptr || args_offset % sizeof(uint32_t) != 0 || args_size == 0 ||
	        params.index_limit == 0 || params.element_size == 0 || params.primitive_size == 0 ||
	        params.primitive_step == 0 || params.primitives_per_group == 0);
	const auto alignment = std::max<uint64_t>(m_graphics.StorageMinAlignment(), 16);
	const auto [mapped, offset] = m_ring.Map(IndirectBytes, alignment);
	EXIT_IF(mapped == nullptr);
	m_ring.Commit();

	// The game's dispatch wrote the arguments earlier in the stream.
	vk::MemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const auto args_binding = Common::AlignDown(args_offset, m_graphics.StorageMinAlignment());
	const vk::DescriptorBufferInfo infos[] {
	    {args, args_binding, args_offset - args_binding + args_size},
	    {m_ring.Handle(), offset, IndirectBytes},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	const auto&            limits = m_graphics.mesh_shader_properties;
	const IndirectPushData push {
	    .args_word            = static_cast<uint32_t>((args_offset - args_binding) / sizeof(uint32_t)),
	    .draw_word            = 0,
	    .command_word         = CommandWord,
	    .index_limit          = params.index_limit,
	    .element_size         = params.element_size,
	    .index_base_low       = static_cast<uint32_t>(params.index_base),
	    .index_base_high      = static_cast<uint32_t>(params.index_base >> 32u),
	    .primitive_size       = params.primitive_size,
	    .primitive_step       = params.primitive_step,
	    .primitives_per_group = params.primitives_per_group,
	    .max_groups_x         = limits.maxMeshWorkGroupCount[0],
	    .max_groups_y         = limits.maxMeshWorkGroupCount[1],
	    .max_groups_total     = limits.maxMeshWorkGroupTotalCount,
	};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch(1, 1, 1);

	vk::MemoryBarrier2 after {};
	after.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	after.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	after.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect |
	                     vk::PipelineStageFlagBits2::eAllGraphics;
	after.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                      vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eMemoryRead;
	dependency.pMemoryBarriers = &after;
	command.pipelineBarrier2(dependency);

	const auto base = m_ring.BufferDeviceAddress() + offset;
	return {.record = base, .command = m_ring.Handle(), .command_offset = offset + RecordBytes};
}

} // namespace Libs::Graphics
