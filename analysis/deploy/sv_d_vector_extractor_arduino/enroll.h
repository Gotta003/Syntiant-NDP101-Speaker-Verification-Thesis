#ifndef ENROLL_H
#define ENROLL_H

#include "global.h"

#define ENROLLMENT_SAMPLES 16
#define BUFFER_SIZE (ENROLLMENT_SAMPLES*D_VECTOR_SIZE*sizeof(float))

// Declare variables as extern
extern float temp_enrollment_buffer[ENROLLMENT_SAMPLES][D_VECTOR_SIZE];
extern uint8_t current_enrollment_count;
extern char current_enrollment_keyword[KEYWORD_LEN];

typedef enum EnrollmentState {
  ENROLL_IDLE,
  ENROLL_KEYWORD_RECEIVED,
  ENROLL_COLLECTING_SAMPLES,
  ENROLL_COMPLETE
} EnrollmentState;

extern EnrollmentState enrollment_state;

// Function declarations
bool startEnrollment(const char* word);
bool addEnrollmentSample(const float* dvector);
bool finalizeEnrollment(void);

#endif