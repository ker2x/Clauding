#include "LockFreeRingBuffer.h"
#include <string.h>

void lfringbuffer_init(LockFreeRingBuffer *rb) {
  if (!rb) {
    return;
  }

  // Note: RING_BUFFER_SIZE must be a power of 2 for mask optimization to work

  // Initialize atomics to zero (required by C11 standard)
  atomic_init(&rb->head, 0);
  atomic_init(&rb->tail, 0);

  // Zero the buffer (defensive - not strictly required since we only read
  // data that has been written, but ensures clean state for debugging)
  memset(rb->buffer, 0, sizeof(rb->buffer));
}

// Write samples to ring buffer (called from audio thread - real-time safe)
// Returns: Number of samples actually written (may be less than 'count' if
// buffer full)
//
// Algorithm:
// 1. Snapshot current head/tail positions atomically
// 2. Calculate free space (keeping 1 slot empty to distinguish full from empty)
// 3. Write data in 1 or 2 chunks (handling circular wrap-around)
// 4. Update head pointer atomically to publish the write
uint32_t lfringbuffer_write(LockFreeRingBuffer *rb, const float *samples,
                            uint32_t count) {
  // Validate inputs
  if (!rb || !samples || count == 0) {
    return 0;
  }

  // Step 1: Load current positions (memory_order_acquire ensures we see latest
  // values) These are unmasked counters that grow indefinitely, wrapping at
  // UINT32_MAX
  // NOTE: ABA problem is not an issue here because we are Single-Producer
  // Single-Consumer (SPSC) and do not use CAS loops. The counters just
  // increment.
  uint32_t currentHead = atomic_load_explicit(&rb->head, memory_order_acquire);
  uint32_t currentTail = atomic_load_explicit(&rb->tail, memory_order_acquire);

  // Step 2: Calculate available space for writing
  // Formula: (tail - head - 1) & mask = free slots
  // We keep 1 slot empty: when head == tail-1, buffer is full (not empty)
  // NOTE: This relies on 2's complement arithmetic. Even if head/tail wrap,
  // (tail - head) correctly calculates the delta in modular arithmetic.
  uint32_t available = (currentTail - currentHead - 1) & (RING_BUFFER_SIZE - 1);

  // Limit write to available space (partial write if buffer is nearly full)
  uint32_t toWrite = count < available ? count : available;

  if (toWrite == 0) {
    return 0; // Buffer full, cannot write anything
  }

  // Step 3: Write samples in one or two chunks (handle wrap-around)
  // Convert unmasked head counter to actual buffer index
  uint32_t writePos = currentHead & (RING_BUFFER_SIZE - 1);

  // Calculate space remaining until end of buffer
  uint32_t firstChunk = RING_BUFFER_SIZE - writePos;

  if (firstChunk >= toWrite) {
    // Case A: All data fits before end of buffer - single contiguous write
    // Example: writePos=100, toWrite=50, buffer[100..149] ← samples[0..49]
    memcpy(&rb->buffer[writePos], samples, toWrite * sizeof(float));
  } else {
    // Case B: Data wraps around end of buffer - two writes needed
    // Example: writePos=4090, toWrite=10
    //   First:  buffer[4090..4095] ← samples[0..5]   (6 samples)
    //   Second: buffer[0..3]       ← samples[6..9]   (4 samples)
    memcpy(&rb->buffer[writePos], samples, firstChunk * sizeof(float));
    memcpy(&rb->buffer[0], &samples[firstChunk],
           (toWrite - firstChunk) * sizeof(float));
  }

  // Step 4: Update head position (memory_order_release ensures writes are
  // visible) CRITICAL: Store unmasked position - let counter wrap naturally at
  // UINT32_MAX Do NOT mask this value! See comment at top of file for
  // explanation.
  // INTENTIONAL OVERFLOW: This addition is allowed to wrap around UINT32_MAX.
  uint32_t newHead = currentHead + toWrite;
  atomic_store_explicit(&rb->head, newHead, memory_order_release);

  return toWrite;
}

// Read samples from ring buffer (called from render thread)
// Returns: Number of samples actually read (may be less than 'count' if buffer
// empty)
//
// Algorithm:
// 1. Snapshot current head/tail positions atomically
// 2. Calculate available samples to read
// 3. Read data in 1 or 2 chunks (handling circular wrap-around)
// 4. Update tail pointer atomically to consume the data
uint32_t lfringbuffer_read(LockFreeRingBuffer *rb, float *samples,
                           uint32_t count) {
  // Validate inputs
  if (!rb || !samples || count == 0) {
    return 0;
  }

  // Step 1: Load current positions (memory_order_acquire ensures we see latest
  // values) These are unmasked counters that grow indefinitely, wrapping at
  // UINT32_MAX
  uint32_t currentHead = atomic_load_explicit(&rb->head, memory_order_acquire);
  uint32_t currentTail = atomic_load_explicit(&rb->tail, memory_order_acquire);

  // Step 2: Calculate available samples for reading
  // Formula: (head - tail) & mask = filled slots
  uint32_t available = (currentHead - currentTail) & (RING_BUFFER_SIZE - 1);

  // Limit read to available samples (partial read if buffer has less than
  // requested)
  uint32_t toRead = count < available ? count : available;

  if (toRead == 0) {
    return 0; // Buffer empty, no data to read
  }

  // Step 3: Read samples in one or two chunks (handle wrap-around)
  // Convert unmasked tail counter to actual buffer index
  uint32_t readPos = currentTail & (RING_BUFFER_SIZE - 1);

  // Calculate space remaining until end of buffer
  uint32_t firstChunk = RING_BUFFER_SIZE - readPos;

  if (firstChunk >= toRead) {
    // Case A: All data available before end of buffer - single contiguous read
    // Example: readPos=200, toRead=100, samples[0..99] ← buffer[200..299]
    memcpy(samples, &rb->buffer[readPos], toRead * sizeof(float));
  } else {
    // Case B: Data wraps around end of buffer - two reads needed
    // Example: readPos=4090, toRead=10
    //   First:  samples[0..5] ← buffer[4090..4095]  (6 samples)
    //   Second: samples[6..9] ← buffer[0..3]        (4 samples)
    memcpy(samples, &rb->buffer[readPos], firstChunk * sizeof(float));
    memcpy(&samples[firstChunk], &rb->buffer[0],
           (toRead - firstChunk) * sizeof(float));
  }

  // Step 4: Update tail position (memory_order_release ensures we won't re-read
  // this data) CRITICAL: Store unmasked position - let counter wrap naturally
  // at UINT32_MAX Do NOT mask this value! See comment at top of file for
  // explanation.
  // INTENTIONAL OVERFLOW: This addition is allowed to wrap around UINT32_MAX.
  uint32_t newTail = currentTail + toRead;
  atomic_store_explicit(&rb->tail, newTail, memory_order_release);

  return toRead;
}
