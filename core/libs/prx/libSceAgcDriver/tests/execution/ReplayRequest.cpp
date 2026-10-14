#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "ControlFlow/include/ControlFlow/RequestSerializer.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<char> ReadFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void Commit(std::uint64_t address, std::size_t size) {
    const auto first = address & ~0xfffull;
    const auto last = (address + size + 0xfffull) & ~0xfffull;
    for (auto page = first; page < last; page += 0x1000ull) {
        MEMORY_BASIC_INFORMATION info{};
        VirtualQuery(reinterpret_cast<void*>(page), &info, sizeof(info));
        if (info.State == MEM_COMMIT) {
            DWORD old = 0;
            VirtualProtect(reinterpret_cast<void*>(page), 0x1000, PAGE_READWRITE, &old);
            continue;
        }
        if (info.State == MEM_RESERVE) {
            auto* block = reinterpret_cast<void*>(page & ~0xffffull);
            VirtualFree(block, 0x10000, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            using Allocate2 = void* (WINAPI*)(HANDLE, void*, SIZE_T, ULONG, ULONG, void*, ULONG);
            static const auto allocate2 = reinterpret_cast<Allocate2>(reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("kernelbase.dll"), "VirtualAlloc2")));
            if (allocate2(GetCurrentProcess(), block, 0x10000, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | MEM_WRITE_WATCH, PAGE_READWRITE, nullptr, 0) == nullptr) throw std::runtime_error("cannot commit guest address " + std::to_string(page) + " error " + std::to_string(GetLastError()));
            continue;
        }
        if (VirtualAlloc(reinterpret_cast<void*>(page & ~0xffffull), 0x10000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) == nullptr) throw std::runtime_error("cannot map guest address " + std::to_string(page) + " error " + std::to_string(GetLastError()));
    }
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::puts("skipped, usage: <shader.req> [groups=X] [load:<hex address>=<file>] [words:<hex address>=<hex word>,...] [dump:<hex address>=<dword count>]...");
        return VulkanTestSkipped;
    }
    try {
        const auto text = ReadFile(argv[1]);
        auto loaded = ShaderRecompiler::RequestSerializer{}.Deserialize(std::string_view(text.data(), text.size()));
        for (const auto& region : loaded.request.context.memory) {
            std::printf("region 0x%llx + 0x%zx\n", static_cast<unsigned long long>(region.guestAddress), region.bytes.size());
            Commit(region.guestAddress, region.bytes.size());
            std::memcpy(reinterpret_cast<void*>(region.guestAddress), region.bytes.data(), region.bytes.size());
        }
        std::uint32_t groups = 1;
        std::uint32_t groupsY = 1;
        for (int i = 2; i < argc; ++i) {
            const std::string argument = argv[i];
            const auto equals = argument.find('=');
            if (argument.rfind("groups=", 0) == 0) {
                char* end = nullptr;
                groups = static_cast<std::uint32_t>(std::strtoul(argument.c_str() + 7, &end, 0));
                if (*end == ',') groupsY = static_cast<std::uint32_t>(std::strtoul(end + 1, nullptr, 0));
            } else if (argument.rfind("load:", 0) == 0) {
                const auto address = std::strtoull(argument.substr(5, equals - 5).c_str(), nullptr, 16);
                const auto bytes = ReadFile(argument.c_str() + equals + 1);
                Commit(address, bytes.size());
                std::memcpy(reinterpret_cast<void*>(address), bytes.data(), bytes.size());
            } else if (argument.rfind("words:", 0) == 0) {
                auto address = std::strtoull(argument.substr(6, equals - 6).c_str(), nullptr, 16);
                const char* cursor = argument.c_str() + equals + 1;
                while (*cursor != 0) {
                    char* end = nullptr;
                    const auto word = static_cast<std::uint32_t>(std::strtoul(cursor, &end, 16));
                    Commit(address, 4);
                    std::memcpy(reinterpret_cast<void*>(address), &word, 4);
                    for (std::size_t r = 0; r < loaded.request.context.memory.size(); ++r) {
                        auto& region = loaded.request.context.memory[r];
                        if (address >= region.guestAddress && address + 4 <= region.guestAddress + region.bytes.size()) {
                            std::memcpy(const_cast<std::byte*>(region.bytes.data()) + (address - region.guestAddress), &word, 4);
                        }
                    }
                    address += 4;
                    cursor = *end == ',' ? end + 1 : end;
                }
            } else if (argument.rfind("dump:", 0) == 0) {
                Commit(std::strtoull(argument.substr(5, equals - 5).c_str(), nullptr, 16), std::strtoul(argument.c_str() + equals + 1, nullptr, 0) * 4);
            }
        }
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        loaded.request.target = device->Target();
        loaded.request.useCache = false;
        const auto result = ShaderRecompiler::Recompile(loaded.request);
        std::printf("recompiled %zu words, host subgroup %u\n", result.spirv.size(), result.hostSubgroupSize);
        device->Dispatch(result, groups, groupsY, 1, {}, loaded.request.shader.codeAddress);
        device->WaitIdle();
        for (int i = 2; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument.rfind("dump:", 0) != 0) continue;
            const auto equals = argument.find('=');
            const auto address = std::strtoull(argument.substr(5, equals - 5).c_str(), nullptr, 16);
            const auto count = std::strtoul(argument.c_str() + equals + 1, nullptr, 0);
            std::printf("0x%llx:", static_cast<unsigned long long>(address));
            for (unsigned long word = 0; word < count; ++word) {
                std::uint32_t value = 0;
                std::memcpy(&value, reinterpret_cast<void*>(address + word * 4), 4);
                float number = 0;
                std::memcpy(&number, &value, 4);
                std::printf("%s%08x(%g)", word % 4 == 0 ? "\n  " : " ", value, number);
            }
            std::puts("");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
