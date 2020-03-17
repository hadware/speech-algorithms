import numpy as np

def voiced_signal(Ams, choices, pitch_estimates, samples_per_skip, signal_length):
    sv_n = np.zeros(signal_length,dtype = 'complex128')
    prev_theta = []
    for f in range(len(pitch_estimates)):# reconstruct signals between each [fK, (f+1)K]
        start_n = f*samples_per_skip
        end_n = (f+1)*samples_per_skip
        for n in range(start_n,end_n):
            current_theta = []
            if f == len(pitch_estimates) - 1:
                max_m = int(pitch_estimates[f]) - 1
            else:
                max_m = max(int(pitch_estimates[f]),int(pitch_estimates[f+1]))-1
            for m in range(max_m):
                # find w0[n]
                if f == len(pitch_estimates) - 1:
                    w_0 = 2* np.pi /pitch_estimates[f]
                else:
                    w_f = 2* np.pi /pitch_estimates[f]
                    w_f_p_1 = 2* np.pi /pitch_estimates[f+1]
                    w_0 = (f+1- n/samples_per_skip) * w_f + (n/samples_per_skip - f) * w_f_p_1
                # find theta_m[n]
                if n == 0:
                    theta_m = (m+1) * w_0
                elif m >= len(prev_theta):# theta_m is 0 if current pitch period exceeds previous pitch period
                    theta_m = (m+1) * w_0
                else:
                    theta_m = prev_theta[m] + (m+1) * w_0
                current_theta.append(theta_m)
                # find Am[n]
                if f == len(pitch_estimates) - 1:
                    Am_n = Ams[f][m]
                else:
                    #find Am_f
                    if m >= int(pitch_estimates[f])-1: #Am,f is 0 if f+1 frame pitch exceeds f frame pitch
                        Am_f = 0
                    elif choices[f][m] == 0:
                        Am_f = 0
                    else:
                        Am_f = Ams[f][m]
                    #find Am_f+1
                    if m >= int(pitch_estimates[f+1])-1: #Am,f+1 is 0 if f frame pitch exceeds f+1 frame pitch
                        Am_f_p_1 = 0
                    elif choices[f+1][m] == 0:
                        Am_f_p_1 = 0
                    else:
                        Am_f_p_1 = Ams[f+1][m]
                    Am_n = (f+1- n/samples_per_skip) * Am_f + (n/samples_per_skip - f) * Am_f_p_1
                sv_n[n] += Am_n * np.cos(theta_m)

            prev_theta = current_theta
    return sv_n

def unvoiced_signal(fft_frames,samples_per_frame,choices,pitch_estimates,fft_length,signal_length,samples_per_skip):
    Uf_ns = []# inverse fft of the noise distribution
    tot_frames = fft_frames.shape[0]
    s_uv = np.zeros(signal_length,dtype ='complex128')
    for i in range(tot_frames):
        Uf = np.zeros(fft_length,dtype ='complex128')# noise distribution
        fft_frame = fft_frames[i]
        period = pitch_estimates[i]
        w_0 = 2 * np.pi / period
        for m in range(0,int(period)-1):
            if choices[i][m] == 0:
                center_freq = w_0 * (m+1)
                lower_bound = w_0 * (m+1-1/2)
                upper_bound = w_0 * (m+1+1/2)
                lower_index = int(lower_bound / (2*np.pi/fft_length)) + 1
                upper_index = min(int(upper_bound /(2*np.pi/fft_length)),fft_length - 1)
                Sw_w = fft_frame[lower_index:upper_index+1]
                Sw_norm = Sw_w *np.conj(Sw_w)
                var = 1/(upper_index - lower_index) * (np.sum(Sw_norm) - 1/2*(Sw_norm[0] + 1/2*(Sw_norm[-1])))
                for k in range(lower_index,upper_index + 1):
                    Uf[k] = complex(np.random.normal(0,np.sqrt(0.5*np.absolute(var))),np.random.normal(0,np.sqrt(0.5*np.absolute(var))))
        Uf_n = np.fft.ifft(Uf,1024)
        Uf_ns.append(Uf_n[0:samples_per_frame])
    Uf_ns = np.array(Uf_ns)
    for f in range(tot_frames):
        start_n = f*samples_per_skip
        end_n = (f+1)*samples_per_skip
        for n in range(start_n,end_n):
            if f == tot_frames - 1:
                s_uv[n] = Uf_ns[f][n-f*samples_per_skip]
            else:
                s_uv[n] = (f+1- n/samples_per_skip) * Uf_ns[f][n-f*samples_per_skip] + (n/samples_per_skip - f) * Uf_ns[f+1][n-(f+1)*samples_per_skip]
    return s_uv
