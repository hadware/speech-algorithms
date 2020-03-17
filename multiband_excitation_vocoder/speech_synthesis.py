import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile
from scipy import signal
import analysis as ana
import synthesis as syn
rate, unnorm_input = scipy.io.wavfile.read('s5.wav')
input = unnorm_input/64000
signal_length = len(input)
samples_per_frame = int(25/1000*rate)
samples_per_skip = int(10/1000*rate)
fft_length = 1024
### analyze signal
pitch_frames = ana.extract_frame(input,rate,samples_per_frame,samples_per_skip)
fft_frames = ana.fft_frame(pitch_frames, fft_length)
pitch_estimates = ana.estimate_pitch(pitch_frames,samples_per_frame,np.arange(20,91))
Ams, all_choices, error_sum =ana.estimate_amplitude(fft_frames,samples_per_frame,pitch_estimates,fft_length)
refined_Am,refined_all_choices,refined_error_sum,refined_periods = ana.estimate_refined_pitch(fft_frames,samples_per_frame,pitch_estimates,fft_length)
###synthesize signal
sv_n = syn.voiced_signal(refined_Am, refined_all_choices, refined_periods,samples_per_skip,signal_length)
suv_n = syn.unvoiced_signal(fft_frames,samples_per_frame,refined_all_choices,refined_periods,fft_length,signal_length,samples_per_skip)
output = np.real(sv_n+suv_n)
scipy.io.wavfile.write("s5_synthesized.wav",rate,output)

# uncomment the following code if you need to see the plots and spectrograms
# plt.plot(np.arange(298), refined_periods)
# plt.title('Pitch Estimates vs. Frame Index')
# plt.ylabel('Pitch Estimates')
# plt.xlabel('Frame [i]')
# plt.show()
# plt.plot(np.arange(298), refined_error_sum)
# plt.title('Error vs. Frame Index')
# plt.ylabel('Error')
# plt.xlabel('Frame [i]')
# plt.show()
# f, t, Sxx = signal.spectrogram(input, rate)
# plt.pcolormesh(t, f, Sxx)
# plt.title('Original Signal')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
# f, t, Sxx = signal.spectrogram(output, rate)
# plt.pcolormesh(t, f, Sxx)
# plt.title('Synthesized Signal')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
