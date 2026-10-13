#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <thread>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		ReportWaitFailure(result, tick);
	}
	Refresh();
}

void MasterSemaphore::RecordSubmit(uint64_t tick, uint32_t debug_op, uint64_t debug_submit,
                                   uint32_t arg0, uint32_t arg1, uint32_t arg2, uint32_t arg3,
                                   uint64_t arg4, uint64_t shader_hash) noexcept {
	auto& record        = m_submits[tick % SubmitHistory];
	record.tick         = tick;
	record.debug_op     = debug_op;
	record.debug_submit = debug_submit;
	record.arg0         = arg0;
	record.arg1         = arg1;
	record.arg2         = arg2;
	record.arg3         = arg3;
	record.arg4         = arg4;
	record.shader_hash  = shader_hash;
}

void MasterSemaphore::ReportWaitFailure(vk::Result result, uint64_t tick) {
	// Every waiting thread sees the same device loss; one report is enough.
	static std::atomic_flag reported = ATOMIC_FLAG_INIT;
	if (reported.test_and_set()) {
		// The first reporter exits the process; do not cut its report short.
		for (;;) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}

	uint64_t counter = 0;
	if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter) !=
	    vk::Result::eSuccess) {
		counter = KnownGpuTick();
	}
	const auto current = CurrentTick();

	const auto report = [](const char* line) {
		LOGF("%s", line);
		std::printf("%s", line);
	};

	char line[256];
	std::snprintf(line, sizeof(line),
	              "vkWaitSemaphores failed: %s (%d), waiting for tick=%" PRIu64
	              " gpu_tick=%" PRIu64 " current_tick=%" PRIu64 "\n",
	              vk::to_string(result).c_str(), static_cast<int>(result), tick, counter,
	              current);
	report(line);

	// The submit right after gpu_tick is the one the GPU never finished.
	for (uint64_t pending = counter + 1; pending < current; ++pending) {
		const auto& record = m_submits[pending % SubmitHistory];
		if (record.tick != pending) {
			continue;
		}
		std::snprintf(line, sizeof(line),
		              "  in-flight tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
		              " args=%u,%u,%u,%u,0x%016" PRIx64
		              " shader_hash=0x%016" PRIx64 "\n",
		              record.tick, record.debug_op, record.debug_submit, record.arg0,
		              record.arg1, record.arg2, record.arg3, record.arg4, record.shader_hash);
		report(line);
	}
	std::fflush(stdout);
	EXIT("vkWaitSemaphores failed: %s\n", vk::to_string(result).c_str());
}

} // namespace Libs::Graphics
