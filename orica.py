import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

def orica(data, weights=None, onlineWhitening=False, numpass=1, block_ica=8, block_white=8, forgetfac='cooling',
          localstat=None, ffdecayrate=0.6, nsub=0, evalconverg=True, verbose=False):
    """

    :param data:
    :param weights:  [W] initial weight matrix     (default -> eye())
    :param sphering:  ['online'|'offline'] use online RLS whitening method or pre-whitening
    :param numpass:  [N] number of passes over input data
    :param block_ica:  [N] block size for ORICA (in samples)
    :param block_white:  [N] block size for online whitening (in samples)
    :param forgetfac:  ['cooling'|'constant'|'adaptive']
        forgetting factor profiles:
        'cooling': monotonically decreasing, for relatively stationary data
        'constant': constant, for online tracking non-stationary data.
        'adaptive': adaptive based on Nonstatinoarity Index (in dev)
        See reference [2] for more information
    :param localstat:  [f] local stationarity (in number of samples) corresponding to
        constant forgetting factor at steady state
    :param ffdecayrate:  [0<f<1] decay rate of (cooling) forgetting factor (default -> 0.6)
    :param nsub:  [N] number of subgaussian sources in EEG signal (default -> 0)
        EEG brain sources are usually supergaussian
        Subgaussian sources are motstly artifact or noise
    :param evalconverg:  [0|1] evaluate convergence such as Non-Stationarity Index
    :param verbose:  ['on'|'off'] give ascii messages  (default -> 'off')
    :return:
    """

    nChs,nPts = data.shape

    ### strategies and parameters for setting up the forgetting factors ###
    class adaptiveFF: pass

    adaptiveFF.profile = forgetfac
    # pars for cooling ff
    adaptiveFF.gamma = ffdecayrate
    adaptiveFF.lambda_0 = 0.995
    # pars for adaptive ff
    adaptiveFF.decayRateAlpha = 0.02
    adaptiveFF.upperBoundBeta = 1e-3
    adaptiveFF.transBandWidthGamma = 1
    adaptiveFF.transBandCenter = 5
    adaptiveFF.lambdaInitial = 0.1

    ### Evaluate convergence such as Non-Stationarity Index (NSI) ###
    class evalConvergence: pass
    evalConvergence.profile = evalconverg
    # Leaky average value (delta) for computing non-stationarity index (NSI).
    # NSI = norm(Rn), where Rn = (1-delta)*Rn + delta*(I-yf^T).
    evalConvergence.leakyAvgDelta = 0.01
    # Leaky average value (delta) for computing variance of source activation.
    # Var = (1-delta)*Var + delta*variance.
    evalConvergence.leakyAvgDeltaVar = 1e-3

    assert forgetfac == 'cooling' or  forgetfac == 'constant' or forgetfac=='adaptive'

    ### initialize state structure ###
    class state: pass
    if weights is not None:
        state.icaweights = weights
    else:
        state.icaweights = np.eye(nChs)

    if onlineWhitening:
        state.icasphere = np.eye(nChs)

    state.lambda_k      = np.zeros((1,block_ica))   # readout lambda
    state.minNonStatIdx = []
    state.counter       = 0 # time index counter, used to keep track of time for computing lambda



    if not onlineWhitening:
        state.icasphere = 2.0 * npla.inv(spla.sqrtm(float(np.cov(data)))) # find the "sphering" matrix = spher()
        data = state.icasphere * data



def dynamicWhitening(blockdata, dataRange, state, adaptiveFF):
    nPts = blockdata.shape[1]
    if adaptiveFF.profile == 'cooling':
        lambd = genCoolingFF(state.counter+dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0)
        if lambd (1) < adaptiveFF.lambda_const:
            lambd = np.tile(adaptiveFF.lambda_const, (1, nPts))
    elif adaptiveFF.profile == 'constant':
        lambd = np.tile(adaptiveFF.lambda_const, (1, nPts))
    elif adaptiveFF.profile == 'adaptive':
        lambd = np.tile(state.lambda_k[-1],(1,nPts))


    v = state.icasphere * blockdata # pre-whitened data
    median_idx = np.ceil(len(lambd)/2)
    lambda_avg = 1 - lambd[median_idx]   # median lambda
    QWhite = lambda_avg/(1-lambda_avg) + np.trace(v.T * v) / nPts
    state.icasphere = 1/lambda_avg * (state.icasphere - v * v.T / nPts / QWhite * state.icasphere)

    return state

def dynamicOrica(blockdata, state, dataRange, adaptiveFF, evalConvergence, nlfunc):
    [nChs, nPts] = blockdata.shape
    f = np.zeros(nChs, nPts)

    # compute source activation using previous weight matrix
    y = state.icaweights * blockdata

    # choose nonlinear functions for super- vs. sub-gaussian
    if nlfunc is None:
        notkurtsign = np.logical_not(state.kurtsign)
        f[state.kurtsign,:]  = -2 * np.tanh(y[state.kurtsign,:]) # Supergaussian
        f[notkurtsign,:] = 2 * np.tanh(y[notkurtsign,:]) # Subgaussian
    else:
        f = nlfunc(y)

    # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
    if evalConvergence.profile:
        modelFitness = np.eye(nChs) + y * f.T/nPts
        variance = blockdata * blockdata
        if state.Rn is None:
            state.Rn = modelFitness
        else:
            state.Rn = (1 - evalConvergence.leakyAvgDelta) * state.Rn + evalConvergence.leakyAvgDelta * modelFitness
        state.nonStatIdx = npla.norm(state.Rn, 'fro')

    if adaptiveFF.profile == 'cooling':
        state.lambda_k = genCoolingFF(state.counter + dataRange, adaptiveFF.gamma, adaptiveFF.lambda_0)
        if state.lambda_k(1) < adaptiveFF.lambda_const
            state.lambda_k = np.tile(adaptiveFF.lambda_const, (1, nPts))
        state.counter = state.counter + nPts

    elif adaptiveFF.profile == 'constant':
        state.lambda_k = np.tile(adaptiveFF.lambda_const,(1,nPts))
    elif adaptiveFF.profile == 'adaptive':
        if state.minNonStatIdx is None:
            state.minNonStatIdx = state.nonStatIdx
        state.minNonStatIdx = max(min(state.minNonStatIdx, state.nonStatIdx),1)
        ratioOfNormRn = state.nonStatIdx/state.minNonStatIdx
        state.lambda_k = genAdaptiveFF(dataRange,state.lambda_k,adaptiveFF.decayRateAlpha,adaptiveFF.upperBoundBeta,adaptiveFF.transBandWidthGamma,adaptiveFF.transBandCenter,ratioOfNormRn);


    # update weight matrix using online recursive ICA block update rule
    lambda_prod = np.prod(1. / (1 - state.lambda_k))
    Q = 1 + state.lambda_k * (np.dot(f, y, axis=1) - 1)
    state.icaweights = lambda_prod * (state.icaweights - y * np.diag(state.lambda_k / Q) * f.T * state.icaweights);

    #
    [V, D] = npla.eig(state.icaweights * state.icaweights.T)
    state.icaweights = V / np.sqrt(D) * V.T * state.icaweights

    return state

def genCoolingFF(t,gamma,lambda_0):
    # lambda = lambda_0 / sample^gamma
    lambd = lambda_0 / np.power(t,gamma)
    return lambd

def genAdaptiveFF(dataRange,lambd,decayRateAlpha,upperBoundBeta,transBandWidthGamma,transBandCenter,ratioOfNormRn):
    # lambda = lambda - DecayRate*lambda + UpperBound*Gain*lambda^2
    # Gain(z) ~ tanh((z/z_{min} - TransBandCenter) / TransBandWidth)
    gainForErrors = upperBoundBeta*0.5*(1+np.tanh((ratioOfNormRn-transBandCenter)/transBandWidthGamma))
    f = lambda n : np.power(1+gainForErrors,n) * lambd[-1] - \
                   decayRateAlpha*(np.power(1+gainForErrors,(2*n-1))-np.power((1+gainForErrors),(n-1))) / np.power(gainForErrors*lambd[-1],2)
    lambd = f(np.arange(len(dataRange)))
    return lambd