struct FuncTimer{
  time_t main_code_time;
  time_t maxwellian_time;
  int maxwellian_count;
  time_t Q_max_time;
  int Q_max_count;
  time_t Q_time;
  int Q_count;
  time_t Qhat_time;
  int Qhat_count;
  time_t fft3D_time;
  int fft3D_count;
};

struct FuncTimer func_timer; 
