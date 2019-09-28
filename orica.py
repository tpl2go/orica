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
    :param verbose:  ['on'|'off'] give ascii messages  (default  -> 'off')
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

def get_WhiteningMatrix(blockdata):
    num_antenna, num_samples = blockdata.shape
    AutocorrelationMatrix = np.matmul(blockdata, np.conj(blockdata.T)) / num_samples
    D, U = npla.eig(AutocorrelationMatrix)

    return np.matmul(np.diag(1./np.sqrt(D)), np.conj(U.T))

def Whitten(blockdata):
    """

    :param blockdata:  shape == (num_antenna, num_samples)
    :return:
    """
    blockdata-=blockdata.mean(1, keepdims=True)
    n,T = blockdata.shape
    m = n

    if m<n:
        #assumes white noise
        D,U     = npla.eig((blockdata.dot(blockdata.conj().T)) / T)
        k       = np.argsort(D)
        puiss   = D[k]
        ibl     = np.sqrt(puiss[-m:]-puiss[:-m].mean())
        bl      = 1/ibl
        W       = np.diag(bl).dot(U[:,k[-m:]].conj().T)
        IW      = U[:,k[-m:]].dot(np.diag(ibl))
    else:
        #assumes no noise
        IW      = spla.sqrtm((blockdata.dot(blockdata.conj().T)) / T)
        W       = npla.inv(IW)

    Y    = W.dot(blockdata)
    return Y

def AppyAndUpdateWhiteningMatrix(blockdata, icasphere, forgettingfactor):
    """
    Online Recursive Independent Component Analysis for Real-time
        Source Separation of High-density EEG

    :param icasphere: shape = (num_antenna, num_antenna)
    :param blockdata:  shape = (num_antenna, num_samples)
    :param forgettingfactor:  scalar between 0 and 1
        as forgettingfactor approaches 1, icaweight never updates, new information is irrelevant
        as forgettingfactor approaches 0, icaweight doesnt converge and learning becomes unstable
    :return:
    """
    whiten_blockdata = np.matmul(icasphere, blockdata)
    num_antenna, num_samples = whiten_blockdata.shape
    AutocorrelationMatrix = np.matmul(whiten_blockdata, np.conj(whiten_blockdata.T))/num_samples
    QWhite = forgettingfactor/ (1. - forgettingfactor) + AutocorrelationMatrix
    icasphere = 1. / forgettingfactor * (icasphere - np.matmul(np.average(np.conj(whiten_blockdata)*whiten_blockdata) / QWhite, icasphere))
    return whiten_blockdata, icasphere


def AppyAndUpdateWhiteningMatrix_Cardoso(blockdata, icasphere , learningrate):
    """
    Equivariant adaptive source separation

    :param icasphere: shape = (num_antenna, num_antenna)
    :param blockdata:  shape = (num_antenna, num_samples)
    :param learningrate:  scalar between 0 and 1
        as forgettingfactor approaches 0, icaweight never updates, new information is irrelevant
        as forgettingfactor approaches 1, learning becomes unstable
    :return:
    """

    num_antenna, num_samples = blockdata.shape
    assert icasphere.shape == (num_antenna, num_antenna)

    whiten_blockdata = np.matmul(icasphere, blockdata)  # shape == (num_antenna, num_samples)
    AutocorrelationMatrix = np.matmul(whiten_blockdata, np.conj(whiten_blockdata.T))/num_samples  # shape == (num_antenna, num_antenna)
    icasphere = (1+learningrate)* icasphere - learningrate * np.matmul(AutocorrelationMatrix,icasphere)

    return whiten_blockdata, icasphere

def ApplyAndUpdateUnmixingMatrix(blockdata, icaweights, lambda_k, nlfunc=None):
    # update weight matrix using online recursive ICA block update rule

    num_antenna, num_samples = blockdata.shape
    assert icaweights.shape == (num_antenna, num_antenna)

    # compute source activation using previous weight matrix
    unmixed_blockdata = np.matmul(icaweights, blockdata)

    # choose nonlinear functions for super- vs. sub-gaussian
    if nlfunc is None:
        # by default, the first half of the outputs are subgaussian sources,
        # the other half are supergaussian sources
        f_blockdata = np.zeros(num_antenna, num_samples)
        num_supergaussians = num_antenna // 2
        f_blockdata[:num_supergaussians,:]  = -2 * np.tanh(unmixed_blockdata[:num_supergaussians,:])  # Supergaussian
        f_blockdata[num_supergaussians:,:] = np.tanh(unmixed_blockdata[num_supergaussians:,:]) - unmixed_blockdata[num_supergaussians:,:] # Subgaussian

    elif type(nlfunc) is list:
        f_blockdata = np.zeros(num_antenna, num_samples)
        assert len(nlfunc) == num_antenna
        for i, fn in enumerate(nlfunc):
            f_blockdata[i,:]  = fn(unmixed_blockdata[i,:])

    else:
        f_blockdata = nlfunc(unmixed_blockdata)

    lambda_prod = np.prod(1. / (1 - lambda_k))
    denominator = (1-lambda_k)/lambda_k  + (np.sum(np.conj(f_blockdata)*unmixed_blockdata, axis=1))
    assert denominator.size == num_samples
    ExpectedOuterProduct = np.einsum('ij,kj->ikj', unmixed_blockdata, np.conj(f_blockdata))  # shape == (num_antenna, num_antenna, num_samples)
    fractions = np.sum(ExpectedOuterProduct/denominator,axis=2)  # shape == (num_antenna, num_antenna)
    icaweights = lambda_prod * (icaweights -  np.matmul(fractions, icaweights))

    # orthogonalize matrix
    [V, D] = npla.eig(np.matmul(icaweights, np.conj(icaweights.T)))
    icaweights = V / np.sqrt(D) * V.T * icaweights

    return unmixed_blockdata, icaweights


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