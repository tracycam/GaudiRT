#include <hlthunk.h>
#include <cstdio>

int main() {
    for (unsigned module = 0; module < 8; ++module) {
        int fd = hlthunk_open_control_by_module_id(module);
        if (fd < 0) { std::printf("module=%u open=%d\n", module, fd); return 1; }
        hlthunk_hw_ip_info info{};
        int rc = hlthunk_get_hw_ip_info(fd, &info);
        std::printf("module=%u rc=%d sram_base=0x%llx sram_bytes=%u dram_bytes=%llu "
                    "tpc_mask=0x%llx mme_mask=0x%x sched_mask=0x%x nic_mask=0x%llx external_nic_mask=0x%llx\n",
                    module, rc, (unsigned long long)info.sram_base_address, info.sram_size,
                    (unsigned long long)info.dram_size, (unsigned long long)info.tpc_enabled_mask_ext,
                    info.mme_enabled_mask, info.sched_arc_enabled_mask,
                    (unsigned long long)info.nic_ports_mask, (unsigned long long)info.nic_ports_external_mask);
        hlthunk_close(fd);
        if (rc) return 2;
    }
}
