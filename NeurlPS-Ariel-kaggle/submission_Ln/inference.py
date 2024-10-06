# search for correct answer
def phase_detector(signal):
    MIN = np.argmin(signal[30:140])+30
    signal1 = signal[:MIN ]
    signal2 = signal[MIN :]

    first_derivative1 = np.gradient(signal1)
    first_derivative1 /= first_derivative1.max()
    first_derivative2 = np.gradient(signal2)
    first_derivative2 /= first_derivative2.max()

    phase1 = np.argmin(first_derivative1)
    phase2 = np.argmax(first_derivative2) + MIN

    return phase1, phase2

def objective(s):
    best_q = 1e10
    for i in range(4) :
        delta = 2
        x = list(range(signal.shape[0]-delta*4))
        y = signal[:p1-delta].tolist() + (signal[p1+delta:p2 - delta] * (1 + s)).tolist() + signal[p2+delta:].tolist()
        
        z = np.polyfit(x, y, deg=i)
        p = np.poly1d(z)
        q = np.abs(p(x) - y).mean()
    
    if q < best_q :
        best_q = q
    
    return q