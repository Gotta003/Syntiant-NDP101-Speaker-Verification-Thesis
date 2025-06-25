#include "enroll.h"

float temp_enrollment_buffer[ENROLLMENT_SAMPLES][D_VECTOR_SIZE] = {0};
uint8_t current_enrollment_count = 0;
char current_enrollment_keyword[KEYWORD_LEN] = {0};
EnrollmentState enrollment_state = ENROLL_IDLE;

bool startEnrollment(const char* word) {
  if(enrollment_state!=ENROLL_IDLE) return false;

  strncpy(current_enrollment_keyword, word, KEYWORD_LEN);
  current_enrollment_count=0;
  enrollment_state=ENROLL_KEYWORD_RECEIVED;
  return true;
}

bool addEnrollmentSample(const float* dvector) {
  if(enrollment_state != ENROLL_KEYWORD_RECEIVED && enrollment_state!=ENROLL_COLLECTING_SAMPLES) {
    return false;
  }
  if(current_enrollment_count>=ENROLLMENT_SAMPLES) {
    return false;
  }
  memcpy(temp_enrollment_buffer[current_enrollment_count], dvector, D_VECTOR_SIZE*sizeof(float));
  current_enrollment_count++;
  enrollment_state=ENROLL_COLLECTING_SAMPLES;
  if(current_enrollment_count==ENROLLMENT_SAMPLES) {
    return finalizeEnrollment();
  }
  return true;
}

bool finalizeEnrollment(void) {
  float mean_vector[D_VECTOR_SIZE]={0};
  for(int s=0; s<ENROLLMENT_SAMPLES; s++) {
    for(int i=0; i<D_VECTOR_SIZE; i++) {
      mean_vector[i]+=temp_enrollment_buffer[s][i];
    }
  }
  for(int i=0; i<D_VECTOR_SIZE; i++) {
    mean_vector[i]/=ENROLLMENT_SAMPLES;
  }
  KeywordIndex* index=findKeywordIndex(current_enrollment_keyword);
  if(index=NULL) {
    enrollment_state=ENROLL_IDLE;
    return false;
  } 
  DVectorEnter* entry=(DVectorEnter*)(memoryUsed+index->start_address);
  entry+=index->vector_count;
  memcpy(entry->dvector, mean_vector, D_VECTOR_SIZE*sizeof(float));
  index->vector_count++;
  enrollment_state=ENROLL_COMPLETE;
  return true;
}

//SPI HANDLE ???
