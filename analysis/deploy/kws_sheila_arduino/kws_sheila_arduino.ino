#include <NDP.h>
#include <NDP_utils.h>
#include <Arduino.h>
//#include "TinyML_init.h"
#include "NDP_init.h"
#include "NDP_loadModel.h"
#include "SAMD21_init.h"
#include "SAMD21_lowpower.h"

#define HIGH_RED() digitalWrite(LED_RED, HIGH)
#define LOW_RED() digitalWrite(LED_RED, LOW)
#define HIGH_BLUE() digitalWrite(LED_BLUE, HIGH)
#define LOW_BLUE() digitalWrite(LED_BLUE, LOW)
#define HIGH_GREEN() digitalWrite(LED_GREEN, HIGH)
#define LOW_GREEN() digitalWrite(LED_GREEN, LOW)

byte FlashType[4]={0, 0, 0, 0};

typedef enum {
  Z_OPENSET,
  SHEILA
} Classifier_match;

typedef enum {
   RED,
   GREEN,
   BLUE,
   YELLOW,
   PINK,
   CYAN,
   WHITE,
   OFF
} Colors;

#define NDP_MICROPHONE 0
#define NDP_SENSOR 1

static volatile Classifier_match s_match;

void led_off() {
  LOW_RED();
  LOW_BLUE();
  LOW_GREEN();
}

void led_color(Colors color) {
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
        Serial.println("LED OFF");
        exit(1);
        break;
  }
  delay(1000);
  led_off();
}

static void ndp_isr(void) {
  led_color(PINK);
  int match_result=NDP.poll();
  Serial.print("Classifier detected: ");
  Serial.println(match_result);

  s_match=(Classifier_match)match_result;
}

void service_ndp() {
  switch(s_match) {
    case Z_OPENSET:
      Serial.println("Unknown or background noise");
      led_color(PINK);
      break;
    case SHEILA:
      Serial.println("Sheila Word Found");
      led_color(RED);
      led_color(BLUE);
      led_color(GREEN);
      led_color(YELLOW);
      led_color(PINK);
      led_color(CYAN);
      break;
    default:
      Serial.println("Uncertain");
      led_color(WHITE);
      break;
  }
}

void syntiant_setup(void) {
  //Memory Ini
  SAMD21_init(0);
  NDP_init("ei_model.bin", NDP_MICROPHONE);
  attachInterrupt(NDP_INT, ndp_isr, HIGH);
  NVMCTRL->CTRLB.bit.SLEEPPRM=NVMCTRL_CTRLB_SLEEPPRM_DISABLED_Val;
  SCB->SCR!=SCB_SCR_SLEEPDEEP_Msk;
}

void syntiant_loop(void) {
  adc_disable();
  usb_serial_disable();
  systick_disable();
  __DSB();
  __WFI();
  systick_enable();
  usb_serial_enable();
  Serial.println("Awake from sleep");
  adc_enable();
  service_ndp();
}

void setup() {
  syntiant_setup();
}

void loop() {
  syntiant_loop();
}
