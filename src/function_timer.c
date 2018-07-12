#include "function_timer.h"

void initialize_func_timer() {
  func_timer.main_code_time = 0;
  func_timer.maxwellian_time = 0;
  func_timer.maxwellian_count = 0;
  func_timer.Q_max_time = 0;
  func_timer.Q_max_count = 0;
  func_timer.Q_time = 0;
  func_timer.Q_count = 0;
  func_timer.Qhat_time = 0;
  func_timer.Qhat_count = 0;
  func_timer.fft3D_time = 0;
  func_timer.fft3D_count = 0;
}
