#ifndef GLOBAL_H
#define GLOBAL_H

#include <stdint.h>
#include <stdbool.h>
#include "stdio.h"

#define D_VECTOR_SIZE 256  
#define WORD_SIZE (D_VECTOR_SIZE*4)
#define MAX_MEMORY (20*1024)
#define SIMILARITY_THRESHOLD 0.8f
#define KEYWORD_NUMBER 1
#define MAX_USERS_PER_KEYWORD ((MAX_MEMORY/WORD_SIZE)/KEYWORD_NUMBER)
#define KEYWORD_LEN 16

#pragma pack(push, 1)

typedef struct DVectorEnter {
 int8_t dvector[D_VECTOR_SIZE];
} DVectorEnter;

typedef struct KeywordIndex {
  char keyword[KEYWORD_LEN];
  uint16_t start_address;
  uint16_t vector_count;
} KeywordIndex;

typedef enum Colors {
  RED,
  GREEN,
  BLUE,
  YELLOW,
  PINK,
  CYAN,
  WHITE,
  OFF
} Colors;

typedef enum Keywords {
  SHEILA,
  NONE
} Keywords;

#pragma pack(pop)

// Declare variables as extern (no memory allocation here)
extern int8_t d_vector[D_VECTOR_SIZE];
extern int8_t memoryUsed[MAX_MEMORY];
extern KeywordIndex keyword_table[KEYWORD_NUMBER];
extern uint16_t current_memory_ptr;
extern volatile bool feature_extraction_done;
extern char* predefined_keywords[KEYWORD_NUMBER];
#endif
