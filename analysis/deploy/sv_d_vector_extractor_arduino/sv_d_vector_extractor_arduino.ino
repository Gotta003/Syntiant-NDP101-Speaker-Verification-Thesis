#include <NDP.h>
#include <NDP_utils.h>
#include <Arduino.h>
#include "NDP_init.h"
#include "SAMD21_init.h"
#include "SAMD21_lowpower.h"
#include "math.h"
#include "enroll.h"
#include "database.h"
//#include "syntiant_fix.h"

#define HIGH_RED() digitalWrite(LED_RED, HIGH)
#define LOW_RED() digitalWrite(LED_RED, LOW)
#define HIGH_BLUE() digitalWrite(LED_BLUE, HIGH)
#define LOW_BLUE() digitalWrite(LED_BLUE, LOW)
#define HIGH_GREEN() digitalWrite(LED_GREEN, HIGH)
#define LOW_GREEN() digitalWrite(LED_GREEN, LOW)

byte FlashType[4]={0, 0, 0, 0};
int8_t d_vector[D_VECTOR_SIZE] = {0};
int8_t memoryUsed[MAX_MEMORY] = {0};
KeywordIndex keyword_table[KEYWORD_NUMBER] = {0};
uint16_t current_memory_ptr = 0;
char* predefined_keywords[KEYWORD_NUMBER] = {
    "sheila"
};
volatile bool feature_extraction_done = false;

Keywords map_words(char* word) {
  if(strcmp(word, "sheila")==0) {
    return SHEILA;
  }
  return NONE;
}

void initDatabase() {
  Serial.println("DEBUG: Initializing database...");

  uint16_t bytes_per_word=MAX_MEMORY/KEYWORD_NUMBER;
  uint16_t vectors_per_word=bytes_per_word/sizeof(DVectorEnter);

  Serial.print("DEBUG: Bytes per word: "); Serial.println(bytes_per_word);
  Serial.print("DEBUG: Vectors per word: "); Serial.println(vectors_per_word);

  for(int i=0; i<KEYWORD_NUMBER; i++) {
    strncpy(keyword_table[i].keyword, predefined_keywords[i], KEYWORD_LEN);
    keyword_table[i].start_address=i*bytes_per_word;
    keyword_table[i].vector_count=0;
    
    Serial.print("DEBUG: Keyword '"); Serial.print(keyword_table[i].keyword);
    Serial.print("' assigned to address: "); Serial.println(keyword_table[i].start_address);

    switch(map_words(keyword_table[i].keyword)) {
      case SHEILA: {
          int counted=(int)(sizeof(sheila_word)/D_VECTOR_SIZE);
          for(int j=0; j<counted; j++) {
              for(int k=0; k<D_VECTOR_SIZE; k++) {
                  memoryUsed[keyword_table[i].start_address+j*D_VECTOR_SIZE+k]=sheila_word[j][k];
              }
          }
          keyword_table[i].vector_count=counted;
          break;
      }
      default:
          Serial.print("ERROR");
          break;
    }
  }
  current_memory_ptr=MAX_MEMORY;//KEYWORD_COUNT*bytes_per_word;
  Serial.print("DEBUG: Current memory pointer set to: "); Serial.println(current_memory_ptr);
  
}

KeywordIndex* findKeywordIndex(const char* word) {
  Serial.print("DEBUG: Searching for keyword '"); Serial.print(word); Serial.println("'");
  for(int i=0; i<KEYWORD_NUMBER; i++) {
    if(strcmp(word, keyword_table[i].keyword)==0) {
      Serial.print("DEBUG: Found keyword at index "); Serial.println(i);
      return &keyword_table[i];
    }
  }
  Serial.println("DEBUG: Keyword not found");
  return NULL;
}

float cosineSimilarity(const int8_t vec1[D_VECTOR_SIZE], const int8_t vec2[D_VECTOR_SIZE]) {  
  Serial.println("DEBUG: Calculating cosine similarity");
  float dot_product=0.0f;
  float norm_vec1=0.0f;
  float norm_vec2=0.0f;
  for(int i=0; i<D_VECTOR_SIZE; i++) {
    dot_product+=vec1[i]*vec2[i];
    norm_vec1+=vec1[i]*vec1[i];
    norm_vec2+=vec2[i]*vec2[i];
  }
  norm_vec1=sqrtf(norm_vec1);
  norm_vec2=sqrtf(norm_vec2);
  if(norm_vec1==0 || norm_vec2==0) {
    Serial.println("DEBUG: Zero vector detected, returning 0 similarity");
    return 0.0f;
  }
  float similarity = dot_product/(norm_vec1*norm_vec2);
  Serial.print("DEBUG: Similarity score: "); Serial.println(similarity, 6);
  return similarity;
}

bool verifyAgainstKeyword(const char* keyword, const int8_t* input_vector) {
  Serial.print("DEBUG: Verifying against keyword '"); Serial.print(keyword); Serial.println("'");
  KeywordIndex* index=findKeywordIndex(keyword);
  if(!index) {
    Serial.println("DEBUG: Keyword index not found");
    return false;
  }
  if(index->vector_count==0) {
    Serial.println("DEBUG: No vectors stored for this keyword");
    return false;
  }

  float best_score=0.0f;
  DVectorEnter* entry=(DVectorEnter*)(memoryUsed+index->start_address);
  Serial.print("DEBUG: Checking "); Serial.print(index->vector_count); Serial.println(" stored vectors");
  
  for(uint16_t i=0; i<index->vector_count; i++) {
    float score=cosineSimilarity(input_vector, entry[i].dvector);
    if(score>SIMILARITY_THRESHOLD) {
      Serial.print("DEBUG: Match found with score "); Serial.println(score, 6);
      return true;
    }
  }
  Serial.println("DEBUG: No matches found above threshold");
  return false;
}

#define NDP_MICROPHONE 0
#define NDP_SENSOR 1

#define DEBUG_LEVEL 2  // 0=off, 1=basic, 2=verbose

#if DEBUG_LEVEL > 0
#define DEBUG_PRINT(x) Serial.print(x)
#define DEBUG_PRINTLN(x) Serial.println(x)
#else
#define DEBUG_PRINT(x)
#define DEBUG_PRINTLN(x)
#endif

#if DEBUG_LEVEL > 1
#define VERBOSE_PRINT(x) Serial.print(x)
#define VERBOSE_PRINTLN(x) Serial.println(x)
#else
#define VERBOSE_PRINT(x)
#define VERBOSE_PRINTLN(x)
#endif

void checkNDPStatus() {
  DEBUG_PRINTLN("\n--- NDP Status Check ---");
  DEBUG_PRINT("NDP initialized: "); DEBUG_PRINTLN(NDP.isInitialized() ? "Yes" : "No");
  DEBUG_PRINT("NDP running: "); DEBUG_PRINTLN(NDP.isRunning() ? "Yes" : "No");
  DEBUG_PRINT("Last error: "); DEBUG_PRINTLN(NDP.getLastError());
  DEBUG_PRINTLN("-----------------------");
}

void led_off() {
  Serial.println("DEBUG: Turning all LEDs off");
  LOW_RED();
  LOW_BLUE();
  LOW_GREEN();
}

void led_color(Colors color) {
  Serial.print("DEBUG: Setting LED color to ");
  switch(color) {
      case RED:
        Serial.println("RED");
        break;
      case BLUE:
        Serial.println("BLUE");
        break;
      case GREEN:
        Serial.println("GREEN");
        break;
      case YELLOW:
        Serial.println("YELLOW");
        break;
      case PINK:
        Serial.println("PINK");
        break;
      case CYAN:
        Serial.println("CYAN");
        break;
      case WHITE:
        Serial.println("WHITE");
        break;
      default:
        Serial.println("UNKNOWN");
        break;
  }
  
  led_off();
  switch(color) {
      case RED:
        HIGH_RED();
        break;
      case BLUE:
        HIGH_BLUE();
        break;
      case GREEN:
        HIGH_GREEN();
        break;
      case YELLOW:
        HIGH_RED();
        HIGH_GREEN();
        break;
      case PINK:
        HIGH_RED();
        HIGH_BLUE();
        break;
      case CYAN:
        HIGH_GREEN();
        HIGH_BLUE();
        break;
      case WHITE:
        HIGH_RED();
        HIGH_GREEN();
        HIGH_BLUE();
        break;
      default:
        break;
  }
}

void dvector_processing() {
  led_color(YELLOW);
  if(!feature_extraction_done) {
    Serial.println("DEBUG: No new features to process");
    return;
  }
  if (feature_extraction_done) {
    Serial.println("DEBUG: Processing extracted features");
      adc_disable();
    usb_serial_disable();
    systick_disable();
    // Process the extracted D-vector here
    // For example, you might want to:
    // 1. Compare with stored speaker profiles
    // 2. Transmit the vector over serial/USB
    // 3. Perform further processing
    
    // Visual feedback
    led_color(YELLOW);
    delay(10000);
    
    led_off();
    feature_extraction_done = false;
    Serial.println("DEBUG: Feature processing complete, flag cleared");
    attachInterrupt(NDP_INT, ndp_isr, HIGH);
  }
}

void relu(int8_t &value) {
    if(value<0) {
      value=0;
    }
}

static void ndp_isr(void) {
  led_color(PINK);
  DEBUG_PRINTLN("\nINTERRUPT FIRED");
  int8_t* dvector_data = NULL;

  if(NDP.pollExtractionComplete(dvector_data)==1) {
     led_color(GREEN);
      DEBUG_PRINTLN("Feature extraction complete!");
      VERBOSE_PRINTLN("D-vector elements:");
      VERBOSE_PRINT("[");
      
      for(int i=0; i<D_VECTOR_SIZE; i++) {
        relu(dvector_data[i]);
        VERBOSE_PRINT(dvector_data[i]);
        if(i<D_VECTOR_SIZE-1) VERBOSE_PRINT(", ");
        if((i+1)%8==0) VERBOSE_PRINTLN();
      }
      VERBOSE_PRINTLN("]");
      
      memcpy(d_vector, dvector_data, D_VECTOR_SIZE*sizeof(int8_t));
      free(dvector_data);
      feature_extraction_done = true;
      DEBUG_PRINTLN("D-vector copied and ready for processing");
  
      // Immediate verification for debugging
      DEBUG_PRINTLN("Running immediate verification...");
      if(verifyAgainstKeyword("sheila", d_vector)) {
        DEBUG_PRINTLN("VERIFICATION SUCCESS!");
        led_color(GREEN);
      } else {
        DEBUG_PRINTLN("Verification failed");
        led_color(RED);
      }
      delay(1000);
      led_off();
      dvector_processing();
  } else {
    DEBUG_PRINTLN("Extraction not complete yet");
    delay(5000);
  }
}

void syntiant_setup(void) {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  DEBUG_PRINTLN("\nStarting Syntiant setup...");
  initDatabase();
  // Check hardware connections first
  DEBUG_PRINTLN("Checking hardware...");
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(500);
  digitalWrite(LED_BUILTIN, LOW);
  DEBUG_PRINTLN("Built-in LED test complete");

  SAMD21_init(0);
  DEBUG_PRINTLN("SAMD21 initialized");

  // Verify NDP initialization
  DEBUG_PRINT("Initializing NDP with model: ei_model.bin... ");
  if(NDP_init("ei_model.bin", NDP_MICROPHONE)==1) {
    DEBUG_PRINTLN("Success!");
  } else {
    DEBUG_PRINTLN("FAILED!");
    while(1) {
      led_color(RED);
      delay(200);
      led_off();
      delay(200);
      DEBUG_PRINTLN("NDP init failed - check model file and connections");
    }
  }
  checkNDPStatus();

  // Test interrupt pin connection
  //pinMode(NDP_INT, INPUT_PULLUP);
  //DEBUG_PRINT("NDP_INT pin state: "); DEBUG_PRINTLN(digitalRead(NDP_INT));
  
  DEBUG_PRINT("Attaching interrupt... ");
  attachInterrupt(NDP_INT, ndp_isr, HIGH);  // Changed to RISING for debugging
  DEBUG_PRINTLN("Done");

  // Additional NDP configuration checks
  DEBUG_PRINT("Microphone enabled: "); 
  DEBUG_PRINTLN(NDP.isMicrophoneEnabled() ? "Yes" : "No");
  
  NVMCTRL->CTRLB.bit.SLEEPPRM = NVMCTRL_CTRLB_SLEEPPRM_DISABLED_Val;
  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
  
  DEBUG_PRINTLN("System ready. Say the wake word!");
  led_color(BLUE);
}

void syntiant_loop(void) {
  static unsigned long last_status = 0;
  
  if(millis() - last_status > 5000) {
    last_status = millis();
    DEBUG_PRINTLN("System heartbeat...");
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
    checkNDPStatus();
  }

  //DEBUG_PRINTLN("Entering low-power mode...");
  /*adc_disable();
  usb_serial_disable();
  systick_disable();
  __DSB();
  __WFI();
  DEBUG_PRINTLN("Woke from low-power mode");
  systick_enable();
  usb_serial_enable();
  adc_enable();*/
}

void setup() {
  syntiant_setup();
}

void loop() {
  syntiant_loop();
}
