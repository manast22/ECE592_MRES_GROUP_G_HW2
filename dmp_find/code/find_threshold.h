#ifndef FIND_THRESHOLD_H
#define FIND_THRESHOLD_H
#include <stdint.h>

extern uint64_t threshold; //below this is cache hit gets set by calibrate()
extern uint64_t timer_ovh; //cycles the timing code costs by itself

int calibrate(void); //sets threshold and returns -1 if hits and misses overlap
uint64_t access_time(volatile uint8_t *p); //cycles to read 1 byte minus overhead

#endif
