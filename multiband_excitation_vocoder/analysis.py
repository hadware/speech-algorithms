import numpy as np

# This function extracts the frames
# samples_per_frame and samples_per_skip in seconds
def extract_frame(input,rate,samples_per_frame,samples_per_skip,discard_last = 1):
    length = len(input)
    pitch_frames = []
    current_sample = 0
    while(current_sample + samples_per_frame <= length):
        current_frame = input[current_sample:(current_sample + samples_per_frame)]
        pitch_frames.append(current_frame)
        current_sample = current_sample + samples_per_skip
    if(current_sample < length):
        if(discard_last == 0):
            current_frame = input[current_sample:]
            pitch_frames.append(current_frame)
    return np.array(pitch_frames)

# This function applies fft to windowed frames
def fft_frame(pitch_frames, fft_length):
    frame_count = pitch_frames.shape[0]
    fft_frames = []
    for f in range(frame_count):
        current_frame = pitch_frames[f] * np.hamming(pitch_frames[f].size)
        fft_frames.append(np.fft.fft(current_frame, fft_length))
    return np.array(fft_frames)

# This function estimates rough pitch integers for a given pitch interval
def estimate_pitch(pitch_frames,samples_per_frame,period_range):
    estimates = []
    for frame in pitch_frames:
        window = np.hamming(samples_per_frame)
        scaled_frame = np.multiply(np.multiply(window,window),frame)
        autocorr = np.correlate(scaled_frame,scaled_frame,'full')
        autocorr_index = np.arange(-samples_per_frame + 1, samples_per_frame)
        rough_estimate = -1
        maximum_theta_p = -np.inf
        for period in period_range:
            min_k = int((-samples_per_frame + 1)/period)
            max_k = int((samples_per_frame - 1)/period)
            theta_p = 0
            for k in range(min_k,max_k + 1):
                theta_p += period * autocorr[np.squeeze(np.where(autocorr_index == k * period))]
            if theta_p > maximum_theta_p:
                maximum_theta_p = theta_p
                rough_estimate = period
        estimates.append(rough_estimate)
    return estimates

# This function estimates amplitude parameters
def estimate_amplitude(fft_frames,samples_per_frame,estimates,fft_length):
    Ams = []
    all_choices = []
    error_sum = []
    for i in range(len(estimates)):
        fft_frame = fft_frames[i]
        period = estimates[i]
        w_0 = 2 * np.pi / period
        Am = []
        choices = []
        em = []
        for m in range(1,int(period)):
            center_freq = w_0 * m
            lower_freq = w_0 * (m-1/2)
            upper_freq = w_0 * (m+1/2)
            center_index = int(np.around(center_freq/(2*np.pi/fft_length)))
            lower_index = int(lower_freq / (2*np.pi/fft_length)) + 1
            upper_index = min(int(upper_freq /(2*np.pi/fft_length)),fft_length - 1)

            S_w = fft_frame[lower_index:upper_index+1]
            fft_window = np.fft.fft(np.hamming(samples_per_frame),fft_length)

            E_w_voiced = np.append(fft_window[fft_length - (center_index - lower_index):fft_length],fft_window[0:upper_index - center_index + 1])# set the excitation for voiced signal to be the window function
            num_Am_voiced = S_w * np.conj(E_w_voiced)
            denom_Am_voiced = E_w_voiced * np.conj(E_w_voiced)
            A_m_voiced = (np.sum(num_Am_voiced))/(np.sum(denom_Am_voiced))
            em_integrand = (S_w - A_m_voiced * E_w_voiced) * np.conj((S_w - A_m_voiced * E_w_voiced))
            em_voiced =  (np.sum(em_integrand))

            E_w_unvoiced = np.ones(len(S_w))# set the excitation for unvoiced signal to be 1
            num_Am_unvoiced = S_w * np.conj(E_w_unvoiced)
            denom_Am_unvoiced = E_w_unvoiced * np.conj(E_w_unvoiced)
            A_m_unvoiced = (np.sum(num_Am_unvoiced))/(np.sum(denom_Am_unvoiced))
            em_integrand_un = (S_w - A_m_unvoiced * E_w_unvoiced) * np.conj((S_w - A_m_unvoiced * E_w_unvoiced))
            em_unvoiced = (np.sum(em_integrand_un))

            if (em_voiced > em_unvoiced):
                choices.append(0)
                em.append(np.absolute(em_unvoiced))
                Am.append(A_m_unvoiced)
            else:
                choices.append(1)
                em.append(np.absolute(em_voiced))
                Am.append(A_m_voiced)
        Ams.append(Am)
        all_choices.append(choices)
        error_sum.append(np.sum(em))
    return Ams,all_choices,error_sum

# estimate refined pitch integers
def estimate_refined_pitch(fft_frames,samples_per_frame,estimates,fft_length):
    refined_Ams = []
    refined_all_choices = []
    refined_error_sum = []
    refined_periods = []
    for i in range(len(estimates)):
        fft_frame = fft_frames[i]
        rough_period = estimates[i]
        periods = np.arange(rough_period-2,rough_period+2+0.2,0.2)
        min_error = np.inf
        refined_period = -1
        refined_Am = []
        refined_choices = []
        refined_em = []
        for period in periods:
            w_0 = 2 * np.pi / period
            Am = []
            choices = []
            em = []
            for m in range(1,int(period)):
                center_freq = w_0 * m
                lower_bound = w_0 * (m-1/2)
                upper_bound = w_0 * (m+1/2)
                center_index = int(np.around(center_freq/(2*np.pi/fft_length)))
                lower_index = int(lower_bound / (2*np.pi/fft_length)) + 1
                upper_index = min(int(upper_bound /(2*np.pi/fft_length)),fft_length - 1)
                # find voiced error
                S_w = fft_frame[lower_index:upper_index+1]
                norm = np.sum(S_w *np.conj(S_w))
                fft_window = np.fft.fft(np.hamming(samples_per_frame),fft_length)
                E_w_voiced = np.append(fft_window[fft_length - (center_index - lower_index):fft_length],fft_window[0:upper_index - center_index + 1])# set the excitation for voiced signal to be the window function
                num_Am_voiced = S_w * np.conj(E_w_voiced)
                denom_Am_voiced = E_w_voiced * np.conj(E_w_voiced)
                A_m_voiced = (np.sum(num_Am_voiced)-1/2*(num_Am_voiced[0]+num_Am_voiced[-1]))/(np.sum(denom_Am_voiced)-1/2*(denom_Am_voiced[0]+denom_Am_voiced[-1]))
                em_integrand = (S_w - A_m_voiced * E_w_voiced) * np.conj((S_w - A_m_voiced * E_w_voiced))
                em_voiced =  (np.sum(em_integrand))-1/2*(em_integrand[0]+em_integrand[-1])
                # find unvoiced error
                E_w_unvoiced = np.ones(len(S_w))# set the excitation for unvoiced signal to be 1
                num_Am_unvoiced = S_w * np.conj(E_w_unvoiced)
                denom_Am_unvoiced = E_w_unvoiced * np.conj(E_w_unvoiced)
                A_m_unvoiced = (np.sum(num_Am_unvoiced)-1/2*(num_Am_unvoiced[0]+num_Am_unvoiced[-1]))/(np.sum(denom_Am_unvoiced)-1/2*(denom_Am_unvoiced[0]+denom_Am_unvoiced[-1]))
                em_integrand_un = (S_w - A_m_unvoiced * E_w_unvoiced) * np.conj((S_w - A_m_unvoiced * E_w_unvoiced))
                em_unvoiced = (np.sum(em_integrand_un) -1/2*(em_integrand_un[0]+em_integrand_un[-1]))
                # label each frame as voiced or unvoiced
                if (em_voiced > em_unvoiced):
                    choices.append(0)
                    em.append(np.absolute(em_unvoiced))
                    Am.append(A_m_unvoiced)
                else:
                    choices.append(1)
                    em.append(np.absolute(em_voiced))
                    Am.append(A_m_voiced)
            current_error_sum = np.sum(em)
            if(current_error_sum < min_error):
                min_error = current_error_sum
                refined_period = period
                refined_Am = Am
                refined_choices = choices
                refined_em = em
        refined_Ams.append(Am)
        refined_all_choices.append(choices)
        refined_error_sum.append(min_error)
        refined_periods.append(refined_period)
    return refined_Ams,refined_all_choices,refined_error_sum,refined_periods
