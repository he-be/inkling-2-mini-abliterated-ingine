// Forced first include on Windows: rpcndr.h (via windows.h) defines `small` as a macro, which breaks
// c10/cuda/CUDACachingAllocator.h (a parameter named `small`). Pull it in once, then drop the macro.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef small
#endif
