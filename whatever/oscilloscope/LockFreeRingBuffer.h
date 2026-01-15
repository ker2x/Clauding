#ifndef LOCKFREE_RINGBUFFER_H
#define LOCKFREE_RINGBUFFER_H

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ring buffer size (must be power of 2)
#define RING_BUFFER_SIZE 4096

// Lock-free single-producer, single-consumer ring buffer
// Uses C11 atomics for thread-safe operation without locks
typedef struct {
  float buffer[RING_BUFFER_SIZE]; // Static array (16KB)
  atomic_uint_fast32_t head;      // Write position (producer)
  atomic_uint_fast32_t tail;      // Read position (consumer)
} LockFreeRingBuffer;

// Initialize ring buffer (cannot fail with static allocation)
void lfringbuffer_init(LockFreeRingBuffer *rb);

// Write samples to ring buffer (called from audio thread)
// Returns number of samples actually written
// REAL-TIME SAFE: No locks, no allocations, no Obj-C
uint32_t lfringbuffer_write(LockFreeRingBuffer *rb, const float *samples,
                            uint32_t count);

// Read samples from ring buffer (called from render thread)
// Returns number of samples actually read
uint32_t lfringbuffer_read(LockFreeRingBuffer *rb, float *samples,
                           uint32_t count);

#ifdef __cplusplus
}
#endif

#endif // LOCKFREE_RINGBUFFER_H
