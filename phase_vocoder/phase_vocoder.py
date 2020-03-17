import numpy as np
from scipy.signal import resample
from scipy.io import wavfile

def windowing(data, rate, frame=0.03, stride=0.01):
    frame_i = 0
    data_length = len(data)
    frame_length = int(frame * rate)
    stride_length = int(2 ** np.floor(np.log2(stride * rate)))
    frame_number = int((data_length - frame_length) / stride_length + 1)
    windowed_frames = np.zeros((frame_number, frame_length))
    for i in range(frame_number):
        curr_frame = data[frame_i:frame_i+np.minimum(frame_length, data_length-(frame_i+frame_length))]
        if len(curr_frame) < frame_length:
            pad_length = frame_length - len(curr_frame)
            curr_frame = np.pad(curr_frame, (int(pad_length/2), pad_length - int(pad_length/2)), 'constant')
        windowed_frames[i] = curr_frame * np.hanning(frame_length)
        frame_i += stride_length
        
    return windowed_frames, frame_length, stride_length

def phase_norm(phase):
    phase += 2 * np.pi
    return phase % (2 * np.pi)

def PV(filename, alpha=1.0):
    '''
    filename: input audio filename
    alpha: time-stretching factor of the original length
    '''
    rate, data = wavfile.read(filename)
    frames, N, Ra = windowing(data, rate)
    Rs = int(Ra * alpha)
    k = np.linspace(0, N/2, N/2+1)
    fft_frames = np.fft.rfft(frames, axis=1)

    ana_delta_pha = phase_norm(k*2*np.pi*Ra/N)
    syn_delta_pha = phase_norm(k*2*np.pi*Rs/N)
    ana_prev_pha = np.angle(fft_frames[0])
    syn_prev_pha = np.angle(fft_frames[0])

    syn_frames = []
    first_mag = np.abs(fft_frames[0])
    first_pha = np.angle(fft_frames[0])
    syn_frames.append((first_mag * np.exp(1j * first_pha)).real)
    for f in fft_frames[1:]:
        ana_curr_pha = phase_norm(np.angle(f).real)
        delta_pha = ana_curr_pha - ana_prev_pha
        delta_pha = phase_norm(delta_pha)
        delta_pha -= ana_delta_pha
        ana_prev_pha = ana_curr_pha
        syn_mag = np.abs(f)
        syn_curr_pha = phase_norm(syn_prev_pha + syn_delta_pha + delta_pha)
        syn_fft = syn_mag * np.exp(1j * syn_curr_pha)
        syn_prev_pha = syn_curr_pha
        syn_f = np.fft.irfft(syn_fft).real
        syn_frames.append(syn_f)
    F, W = frames.shape
    output_length = (F-1) * Rs + W
    output = np.zeros(output_length)
    concat_i = 0
    for syn_f in syn_frames:
        output[concat_i:concat_i+len(syn_f)] += syn_f
        concat_i += Rs
    output /= np.amax(output)
    wavfile.write('output_%.1f.wav'%(alpha), rate, output)

for alpha in np.linspace(0.5, 1.5, num=11):
    PV('input.wav', alpha=alpha)
    
