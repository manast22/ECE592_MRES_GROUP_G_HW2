/* Prime+Probe measurements. Build and run instructions are in README.txt.
 * stdout: samples; stderr: setup, calibration and batch timings. */
#define _GNU_SOURCE
#include <cpuid.h>
#include <inttypes.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

#define MAX_WAYS 64

typedef struct {
    uint64_t a, b, ns;
    int bit, order;
} Sample;

static int ways, line;
static size_t stride;
static uint8_t *memory;

static uint64_t now_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) {
        perror("clock_gettime");
        exit(1);
    }
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

// Fence the load between timestamps.
static uint64_t time_load(volatile uint8_t *address) {
    unsigned int aux;
    _mm_lfence();
    uint64_t start = __rdtsc();
    _mm_lfence();
    (void)*address;
    _mm_lfence();
    uint64_t end = __rdtscp(&aux);
    _mm_lfence();
    return end - start;
}

// Stride = sets * line size, keeping the set index fixed.
static void prime(volatile uint8_t *base) {
    for (int i = 0; i < ways; ++i)
        (void)base[i * stride];
}

static uint64_t probe(volatile uint8_t *base) {
    uint64_t ticks = 0;
    for (int i = 0; i < ways; ++i)
        ticks += time_load(base + i * stride);
    return ticks;
}

// 0: signal, 1: no reads, 2: unrelated sets.
__attribute__((noinline))
static void victim(int bit, int mode) {
    if (mode == 1)
        return;
    int set = mode == 0 ? bit : bit + 2;
    prime(memory + ways * stride + set * line);
}

// Touch recording pages before timing.
static void touch_pages(void *buffer, size_t bytes, size_t page) {
    volatile uint8_t *p = buffer;
    for (size_t offset = 0; offset < bytes; offset += page)
        p[offset] = 0;
    p[bytes - 1] = 0;
}

static uint64_t collect(Sample *samples, int count, int mode, int cpu) {
    // Shuffle balanced labels before timing.
    for (int i = 0; i < count; ++i) {
        samples[i].bit = i % 2;
        samples[i].order = rand() & 1;
    }
    for (int i = count - 1; i > 0; --i) {
        int j = rand() % (i + 1);
        int bit = samples[i].bit;
        samples[i].bit = samples[j].bit;
        samples[j].bit = bit;
    }
    if (sched_getcpu() != cpu) {
        fprintf(stderr, "CPU affinity check failed before batch\n");
        exit(1);
    }
    uint64_t batch_start = now_ns();
    for (int i = 0; i < count; ++i) {
        int bit = samples[i].bit;
        int order = samples[i].order;
        uint64_t start = now_ns();
        prime(memory);
        prime(memory + line);
        victim(bit, mode);
        uint64_t first = probe(memory + order * line);
        uint64_t second = probe(memory + (1 - order) * line);
        uint64_t end = now_ns();
        samples[i].a = order == 0 ? first : second;
        samples[i].b = order == 0 ? second : first;
        samples[i].ns = end - start;
    }
    uint64_t elapsed = now_ns() - batch_start;
    if (sched_getcpu() != cpu) {
        fprintf(stderr, "CPU affinity check failed after batch\n");
        exit(1);
    }
    return elapsed;
}

static int compare_scores(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

static double median(int64_t *scores, int count) {
    qsort(scores, (size_t)count, sizeof(*scores), compare_scores);
    return ((double)scores[(count - 1) / 2] + scores[count / 2]) / 2;
}

static void write_samples(Sample *samples, int count, const char *mode,
                          int run, double threshold) {
    const char *phase = run < 0 ? "calibration" : "evaluation";
    for (int i = 0; i < count; ++i) {
        Sample s = samples[i];
        int64_t score = (int64_t)s.b - (int64_t)s.a;
        int guess = run < 0 ? -1 : score > threshold;
        printf("%s,%s,%d,%d,%d,%d,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
               phase, mode, run, i, s.bit, s.order, s.a, s.b, s.ns, guess);
    }
}

static int parse_int(const char *text, int low, int high) {
    char *end;
    long n = strtol(text, &end, 10);
    if (!*text || *end || n < low || n > high) {
        fprintf(stderr, "Invalid argument: %s\n", text);
        exit(2);
    }
    return (int)n;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 9) {
        fprintf(stderr, "Usage: %s CPU WAYS SETS LINE_BYTES SEED [TRIALS RUNS CAL_PER_BIT]\n", argv[0]);
        return 2;
    }
    int cpu = parse_int(argv[1], 0, CPU_SETSIZE - 1);
    ways = parse_int(argv[2], 1, MAX_WAYS);
    int sets = parse_int(argv[3], 4, 4096);
    line = parse_int(argv[4], 16, 1024);
    unsigned int seed = (unsigned int)parse_int(argv[5], 1, 2147483647);
    srand(seed);
    int trials = argc == 9 ? parse_int(argv[6], 2, 1000000) : 10000;
    int runs = argc == 9 ? parse_int(argv[7], 1, 100) : 10;
    int calibration = argc == 9 ? parse_int(argv[8], 2, 1000000) : 10000;
    stride = (size_t)sets * line;
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    if (trials % 2 || stride > page || (stride & (stride - 1))) {
        fprintf(stderr, "Unsupported geometry or unbalanced trial count\n");
        return 2;
    }
    unsigned int a, b, c, d;
    if (!__get_cpuid(0x80000001U, &a, &b, &c, &d) || !(d & (1U << 27))) {
        fprintf(stderr, "RDTSCP is unavailable\n");
        return 3;
    }
    cpu_set_t allowed, selected;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) || !CPU_ISSET(cpu, &allowed)) {
        fprintf(stderr, "Requested CPU is not in the allowed affinity mask\n");
        return 2;
    }
    CPU_ZERO(&selected);
    CPU_SET(cpu, &selected);
    if (sched_setaffinity(0, sizeof(selected), &selected)) {
        perror("sched_setaffinity");
        return 1;
    }

    uint64_t setup_start = now_ns();
    // First half: attacker. Second half: victim.
    size_t bytes = 2 * (size_t)ways * stride;
    int capacity = trials > 2 * calibration ? trials : 2 * calibration;
    memory = aligned_alloc(page, ((bytes + page - 1) / page) * page);
    Sample *samples = calloc((size_t)capacity, sizeof(*samples));
    int64_t *scores = calloc(2 * (size_t)calibration, sizeof(*scores));
    if (!memory || !samples || !scores) {
        fprintf(stderr, "Allocation failed\n");
        free(memory);
        free(samples);
        free(scores);
        return 1;
    }
    memset(memory, 0, bytes);
    touch_pages(samples, (size_t)capacity * sizeof(*samples), page);
    touch_pages(scores, 2 * (size_t)calibration * sizeof(*scores), page);
    uint64_t setup_ns = now_ns() - setup_start;
    fprintf(stderr, "{\"type\":\"setup\",\"cpu\":%d,\"ways\":%d,\"sets\":%d,\"line_bytes\":%d,\"stride_bytes\":%zu,\"seed\":%u,\"setup_ns\":%" PRIu64 ",\"attack_allocation_bytes\":%zu,\"recording_allocation_bytes\":%zu}\n",
            cpu, ways, sets, line, stride, seed, setup_ns, bytes,
            (size_t)capacity * sizeof(*samples) + 2 * (size_t)calibration * sizeof(*scores));
    puts("phase,mode,run,trial,bit,probe_order,time_a_ticks,time_b_ticks,latency_ns,guess");

    // Calibrate once, then keep the threshold fixed.
    uint64_t calibration_ns = collect(samples, 2 * calibration, 0, cpu);
    int counts[2] = {0, 0};
    for (int i = 0; i < 2 * calibration; ++i) {
        int bit = samples[i].bit;
        scores[bit * calibration + counts[bit]++] = (int64_t)samples[i].b - (int64_t)samples[i].a;
    }
    double median0 = median(scores, calibration);
    double median1 = median(scores + calibration, calibration);
    double threshold = (median0 + median1) / 2;
    fprintf(stderr, "{\"type\":\"calibration\",\"samples_per_bit\":%d,\"median0\":%.6f,\"median1\":%.6f,\"threshold\":%.6f,\"elapsed_ns\":%" PRIu64 "}\n",
            calibration, median0, median1, threshold, calibration_ns);
    fflush(stderr);
    write_samples(samples, 2 * calibration, "signal", -1, threshold);

    const char *modes[] = {"signal", "no_victim", "unrelated"};
    for (int run = 0; run < runs; ++run) {
        // Rotate signal and control order.
        for (int offset = 0; offset < 3; ++offset) {
            int mode = (run + offset) % 3;
            uint64_t elapsed = collect(samples, trials, mode, cpu);
            fprintf(stderr, "{\"type\":\"batch\",\"run\":%d,\"mode\":\"%s\",\"trials\":%d,\"elapsed_ns\":%" PRIu64 ",\"cpu_after\":%d}\n",
                    run, modes[mode], trials, elapsed, sched_getcpu());
            write_samples(samples, trials, modes[mode], run, threshold);
        }
    }
    struct rusage usage;
    if (!getrusage(RUSAGE_SELF, &usage)) {
        fprintf(stderr, "{\"type\":\"resources\",\"max_rss_kib\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld,\"voluntary_context_switches\":%ld,\"involuntary_context_switches\":%ld}\n",
                usage.ru_maxrss, usage.ru_minflt, usage.ru_majflt, usage.ru_nvcsw, usage.ru_nivcsw);
    }
    free(memory);
    free(samples);
    free(scores);
    return 0;
}
