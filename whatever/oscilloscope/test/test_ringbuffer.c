/*
 * Unit Tests for LockFreeRingBuffer
 *
 * Compile: clang -std=c11 -O2 -I.. test_ringbuffer.c ../LockFreeRingBuffer.c -o test_ringbuffer
 * Run: ./test_ringbuffer
 *
 * Tests cover:
 * - Basic initialization
 * - Simple read/write operations
 * - Full/empty buffer conditions
 * - Wrap-around behavior (critical for race condition prevention)
 * - Partial reads/writes
 * - NULL/invalid input handling
 * - High-volume stress test
 */

#include "LockFreeRingBuffer.h"
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test framework macros
#define ANSI_RED "\x1b[31m"
#define ANSI_GREEN "\x1b[32m"
#define ANSI_YELLOW "\x1b[33m"
#define ANSI_RESET "\x1b[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    tests_run++;                                                               \
    printf("  %-50s ", name);                                                  \
  } while (0)

#define PASS()                                                                 \
  do {                                                                         \
    tests_passed++;                                                            \
    printf(ANSI_GREEN "[PASS]" ANSI_RESET "\n");                               \
  } while (0)

#define FAIL(msg)                                                              \
  do {                                                                         \
    printf(ANSI_RED "[FAIL]" ANSI_RESET " %s\n", msg);                         \
  } while (0)

#define ASSERT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      FAIL(msg);                                                               \
      return;                                                                  \
    }                                                                          \
  } while (0)

// ============================================================================
// Test: Initialization
// ============================================================================

void test_init_zeros_head_and_tail(void) {
  TEST("init: head and tail are zero");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  uint32_t head = atomic_load(&rb.head);
  uint32_t tail = atomic_load(&rb.tail);

  ASSERT(head == 0, "head should be 0");
  ASSERT(tail == 0, "tail should be 0");
  PASS();
}

void test_init_null_safe(void) {
  TEST("init: handles NULL pointer");

  // Should not crash
  lfringbuffer_init(NULL);
  PASS();
}

// ============================================================================
// Test: Basic Write/Read
// ============================================================================

void test_write_single_sample(void) {
  TEST("write: single sample");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float sample = 0.5f;
  uint32_t written = lfringbuffer_write(&rb, &sample, 1);

  ASSERT(written == 1, "should write 1 sample");
  PASS();
}

void test_read_single_sample(void) {
  TEST("read: single sample");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float write_sample = 0.75f;
  lfringbuffer_write(&rb, &write_sample, 1);

  float read_sample = 0.0f;
  uint32_t read_count = lfringbuffer_read(&rb, &read_sample, 1);

  ASSERT(read_count == 1, "should read 1 sample");
  ASSERT(fabsf(read_sample - 0.75f) < 0.0001f, "sample value mismatch");
  PASS();
}

void test_write_read_multiple(void) {
  TEST("write/read: multiple samples preserve order");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float write_buf[100];
  for (int i = 0; i < 100; i++) {
    write_buf[i] = (float)i * 0.01f;
  }

  uint32_t written = lfringbuffer_write(&rb, write_buf, 100);
  ASSERT(written == 100, "should write 100 samples");

  float read_buf[100];
  uint32_t read_count = lfringbuffer_read(&rb, read_buf, 100);
  ASSERT(read_count == 100, "should read 100 samples");

  for (int i = 0; i < 100; i++) {
    ASSERT(fabsf(read_buf[i] - write_buf[i]) < 0.0001f, "data corrupted");
  }
  PASS();
}

// ============================================================================
// Test: Empty Buffer
// ============================================================================

void test_read_empty_buffer(void) {
  TEST("read: empty buffer returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float buf[10];
  uint32_t read_count = lfringbuffer_read(&rb, buf, 10);

  ASSERT(read_count == 0, "should return 0 for empty buffer");
  PASS();
}

// ============================================================================
// Test: Full Buffer
// ============================================================================

void test_write_full_buffer(void) {
  TEST("write: full buffer returns partial count");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  // Fill buffer (capacity is RING_BUFFER_SIZE - 1 due to empty slot)
  float buf[RING_BUFFER_SIZE];
  for (uint32_t i = 0; i < RING_BUFFER_SIZE; i++) {
    buf[i] = (float)i;
  }

  uint32_t written = lfringbuffer_write(&rb, buf, RING_BUFFER_SIZE);

  // Should write RING_BUFFER_SIZE - 1 samples (one slot kept empty)
  ASSERT(written == RING_BUFFER_SIZE - 1, "should write size-1 samples");
  PASS();
}

void test_write_to_full_buffer_returns_zero(void) {
  TEST("write: to completely full buffer returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  // Fill buffer
  float buf[RING_BUFFER_SIZE];
  lfringbuffer_write(&rb, buf, RING_BUFFER_SIZE);

  // Try to write more
  float extra = 1.0f;
  uint32_t written = lfringbuffer_write(&rb, &extra, 1);

  ASSERT(written == 0, "should return 0 when full");
  PASS();
}

// ============================================================================
// Test: Wrap-Around (CRITICAL - tests the fixed race condition)
// ============================================================================

void test_wraparound_write_read(void) {
  TEST("wrap-around: data integrity across boundary");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  // Strategy: Fill near end, read some, write across boundary
  // This tests the critical wrap-around logic

  // Step 1: Fill most of the buffer
  float fill_buf[RING_BUFFER_SIZE - 100];
  for (uint32_t i = 0; i < RING_BUFFER_SIZE - 100; i++) {
    fill_buf[i] = (float)i;
  }
  lfringbuffer_write(&rb, fill_buf, RING_BUFFER_SIZE - 100);

  // Step 2: Read most of it back (move tail forward)
  float discard[RING_BUFFER_SIZE - 200];
  lfringbuffer_read(&rb, discard, RING_BUFFER_SIZE - 200);

  // Now head is near end, tail is also near end
  // Step 3: Write data that will wrap around
  float wrap_data[300];
  for (int i = 0; i < 300; i++) {
    wrap_data[i] = 1000.0f + (float)i; // Distinctive values
  }
  uint32_t written = lfringbuffer_write(&rb, wrap_data, 300);
  ASSERT(written == 300, "should write 300 samples across boundary");

  // Step 4: Read remaining old data
  float old_data[100];
  lfringbuffer_read(&rb, old_data, 100);

  // Step 5: Read the wrapped data and verify
  float verify_buf[300];
  uint32_t read_count = lfringbuffer_read(&rb, verify_buf, 300);
  ASSERT(read_count == 300, "should read 300 samples");

  for (int i = 0; i < 300; i++) {
    float expected = 1000.0f + (float)i;
    ASSERT(fabsf(verify_buf[i] - expected) < 0.0001f,
           "wrap-around data corrupted");
  }
  PASS();
}

void test_wraparound_counter_overflow(void) {
  TEST("wrap-around: counter overflow simulation");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  // Simulate counters near UINT32_MAX by manipulating atomics directly
  // This tests the unmasked counter design
  uint32_t near_max = UINT32_MAX - 100;
  atomic_store(&rb.head, near_max);
  atomic_store(&rb.tail, near_max);

  // Write some data (will wrap counter past UINT32_MAX)
  float write_buf[200];
  for (int i = 0; i < 200; i++) {
    write_buf[i] = (float)i * 0.5f;
  }
  uint32_t written = lfringbuffer_write(&rb, write_buf, 200);
  ASSERT(written == 200, "should write across counter overflow");

  // Read it back
  float read_buf[200];
  uint32_t read_count = lfringbuffer_read(&rb, read_buf, 200);
  ASSERT(read_count == 200, "should read across counter overflow");

  // Verify data integrity
  for (int i = 0; i < 200; i++) {
    ASSERT(fabsf(read_buf[i] - write_buf[i]) < 0.0001f,
           "data corrupted at counter overflow");
  }
  PASS();
}

// ============================================================================
// Test: Partial Operations
// ============================================================================

void test_partial_read(void) {
  TEST("partial: read less than available");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float write_buf[100];
  for (int i = 0; i < 100; i++) {
    write_buf[i] = (float)i;
  }
  lfringbuffer_write(&rb, write_buf, 100);

  // Read only 50
  float read_buf[50];
  uint32_t read_count = lfringbuffer_read(&rb, read_buf, 50);

  ASSERT(read_count == 50, "should read exactly 50");
  for (int i = 0; i < 50; i++) {
    ASSERT(fabsf(read_buf[i] - (float)i) < 0.0001f, "partial read corrupted");
  }

  // Read remaining 50
  read_count = lfringbuffer_read(&rb, read_buf, 50);
  ASSERT(read_count == 50, "should read remaining 50");
  for (int i = 0; i < 50; i++) {
    ASSERT(fabsf(read_buf[i] - (float)(i + 50)) < 0.0001f,
           "second partial read corrupted");
  }
  PASS();
}

void test_read_more_than_available(void) {
  TEST("partial: read more than available returns actual");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float write_buf[50];
  lfringbuffer_write(&rb, write_buf, 50);

  float read_buf[100];
  uint32_t read_count = lfringbuffer_read(&rb, read_buf, 100);

  ASSERT(read_count == 50, "should return actual available count");
  PASS();
}

// ============================================================================
// Test: Invalid Input Handling
// ============================================================================

void test_write_null_buffer(void) {
  TEST("invalid: write with NULL buffer returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  uint32_t written = lfringbuffer_write(&rb, NULL, 10);
  ASSERT(written == 0, "should return 0 for NULL buffer");
  PASS();
}

void test_read_null_buffer(void) {
  TEST("invalid: read with NULL buffer returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float sample = 1.0f;
  lfringbuffer_write(&rb, &sample, 1);

  uint32_t read_count = lfringbuffer_read(&rb, NULL, 1);
  ASSERT(read_count == 0, "should return 0 for NULL buffer");
  PASS();
}

void test_write_null_ringbuffer(void) {
  TEST("invalid: write to NULL ringbuffer returns 0");

  float buf[10];
  uint32_t written = lfringbuffer_write(NULL, buf, 10);
  ASSERT(written == 0, "should return 0 for NULL ringbuffer");
  PASS();
}

void test_read_null_ringbuffer(void) {
  TEST("invalid: read from NULL ringbuffer returns 0");

  float buf[10];
  uint32_t read_count = lfringbuffer_read(NULL, buf, 10);
  ASSERT(read_count == 0, "should return 0 for NULL ringbuffer");
  PASS();
}

void test_write_zero_count(void) {
  TEST("invalid: write with count=0 returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float sample = 1.0f;
  uint32_t written = lfringbuffer_write(&rb, &sample, 0);
  ASSERT(written == 0, "should return 0 for zero count");
  PASS();
}

void test_read_zero_count(void) {
  TEST("invalid: read with count=0 returns 0");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  float sample = 1.0f;
  lfringbuffer_write(&rb, &sample, 1);

  float buf[1];
  uint32_t read_count = lfringbuffer_read(&rb, buf, 0);
  ASSERT(read_count == 0, "should return 0 for zero count");
  PASS();
}

// ============================================================================
// Test: Stress Test (High Volume)
// ============================================================================

void test_stress_sequential(void) {
  TEST("stress: 1M samples sequential write/read");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  #define TOTAL_SAMPLES 1000000
  #define CHUNK_SIZE 256

  float write_buf[CHUNK_SIZE];
  float read_buf[CHUNK_SIZE];

  uint32_t total_written = 0;
  uint32_t total_read = 0;

  while (total_read < TOTAL_SAMPLES) {
    // Write a chunk (values are based on total_written counter)
    if (total_written < TOTAL_SAMPLES) {
      uint32_t to_write = CHUNK_SIZE;
      if (total_written + to_write > TOTAL_SAMPLES) {
        to_write = TOTAL_SAMPLES - total_written;
      }
      for (uint32_t i = 0; i < to_write; i++) {
        write_buf[i] = (float)(total_written + i);
      }
      uint32_t written = lfringbuffer_write(&rb, write_buf, to_write);
      total_written += written;
    }

    // Read a chunk and verify values match expected sequence
    uint32_t read_count = lfringbuffer_read(&rb, read_buf, CHUNK_SIZE);
    for (uint32_t i = 0; i < read_count; i++) {
      float expected = (float)(total_read + i);
      if (fabsf(read_buf[i] - expected) > 0.0001f) {
        FAIL("data corruption in stress test");
        return;
      }
    }
    total_read += read_count;
  }

  ASSERT(total_read == TOTAL_SAMPLES, "should process all samples");
  PASS();

  #undef TOTAL_SAMPLES
  #undef CHUNK_SIZE
}

// ============================================================================
// Test: Concurrent Access (Producer-Consumer)
// ============================================================================

typedef struct {
  LockFreeRingBuffer *rb;
  uint32_t sample_count;
  int success;
} ThreadArg;

void *producer_thread(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  uint32_t written = 0;

  while (written < ta->sample_count) {
    float sample = (float)written;  // Use written count as sequence
    uint32_t w = lfringbuffer_write(ta->rb, &sample, 1);
    written += w;
    // Yield occasionally to simulate real-world timing
    if (written % 1000 == 0) {
      sched_yield();
    }
  }

  ta->success = 1;
  return NULL;
}

void *consumer_thread(void *arg) {
  ThreadArg *ta = (ThreadArg *)arg;
  uint32_t read_count = 0;

  while (read_count < ta->sample_count) {
    float sample;
    uint32_t r = lfringbuffer_read(ta->rb, &sample, 1);
    if (r > 0) {
      float expected = (float)read_count;  // Use read count as expected sequence
      if (fabsf(sample - expected) > 0.0001f) {
        ta->success = 0;
        return NULL;
      }
      read_count += r;
    }
    // Yield occasionally
    if (read_count % 1000 == 0) {
      sched_yield();
    }
  }

  ta->success = 1;
  return NULL;
}

void test_concurrent_producer_consumer(void) {
  TEST("concurrent: producer-consumer thread safety");

  LockFreeRingBuffer rb;
  lfringbuffer_init(&rb);

  const uint32_t SAMPLES = 100000;

  ThreadArg producer_arg = {&rb, SAMPLES, 0};
  ThreadArg consumer_arg = {&rb, SAMPLES, 0};

  pthread_t producer, consumer;
  pthread_create(&producer, NULL, producer_thread, &producer_arg);
  pthread_create(&consumer, NULL, consumer_thread, &consumer_arg);

  pthread_join(producer, NULL);
  pthread_join(consumer, NULL);

  ASSERT(producer_arg.success == 1, "producer failed");
  ASSERT(consumer_arg.success == 1, "consumer failed or data corrupted");
  PASS();
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
  printf("\n" ANSI_YELLOW "═══════════════════════════════════════════════════"
                          "═══════════════════════\n");
  printf(" LockFreeRingBuffer Unit Tests\n");
  printf("═══════════════════════════════════════════════════════════════════"
         "════════\n" ANSI_RESET);

  printf("\n[Initialization]\n");
  test_init_zeros_head_and_tail();
  test_init_null_safe();

  printf("\n[Basic Operations]\n");
  test_write_single_sample();
  test_read_single_sample();
  test_write_read_multiple();

  printf("\n[Empty Buffer]\n");
  test_read_empty_buffer();

  printf("\n[Full Buffer]\n");
  test_write_full_buffer();
  test_write_to_full_buffer_returns_zero();

  printf("\n[Wrap-Around (Critical)]\n");
  test_wraparound_write_read();
  test_wraparound_counter_overflow();

  printf("\n[Partial Operations]\n");
  test_partial_read();
  test_read_more_than_available();

  printf("\n[Invalid Input Handling]\n");
  test_write_null_buffer();
  test_read_null_buffer();
  test_write_null_ringbuffer();
  test_read_null_ringbuffer();
  test_write_zero_count();
  test_read_zero_count();

  printf("\n[Stress Tests]\n");
  test_stress_sequential();

  printf("\n[Concurrent Access]\n");
  test_concurrent_producer_consumer();

  printf("\n" ANSI_YELLOW "═══════════════════════════════════════════════════"
                          "═══════════════════════\n" ANSI_RESET);
  if (tests_passed == tests_run) {
    printf(ANSI_GREEN " ✓ All %d tests passed\n" ANSI_RESET, tests_run);
  } else {
    printf(ANSI_RED " ✗ %d/%d tests passed (%d failed)\n" ANSI_RESET,
           tests_passed, tests_run, tests_run - tests_passed);
  }
  printf(ANSI_YELLOW "═══════════════════════════════════════════════════════"
                     "════════════════════\n\n" ANSI_RESET);

  return tests_passed == tests_run ? 0 : 1;
}
