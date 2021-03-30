import jax
import jax.numpy as np
from functools import partial

def default_P(k, A, B):
    return A*k**-B


class powerBoxJax:
    def __init__(self, pk=None, k=None):
        
        if pk is None:
            self.pk = default_P
        else:
            self.pk = pk
            
        if k is None:
            self.k = np.sqrt(np.sum(np.array(np.meshgrid(*(
                (np.hstack((np.arange(0, _shape//2 + 1), 
                    np.arange(-_shape//2 + 1, 0))) * 2*np.pi / _shape)**2
                    for _shape in shape))), axis=0))
        else:
            self.k = k
        
    def simulator(self, rng, θ, simulator_args=simulator_args):
        def P(k, A=1, B=1):
            return self.pk(k, A, B)

        def fn(key, A, B):
            shape = simulator_args["shape"]

            k = self.k

            new_shape = ()
            for _shape in shape:
                if _shape % 2 == 0:
                    new_shape += (_shape+1,)
                else:
                    new_shape += (_shape,)


            key1,key2 = jax.random.split(key)

            foreground = foregrounds[jax.random.randint(key2, 
                                            minval=0, maxval=1000, shape=())]

            # L is in length units, like Gpc
            L = simulator_args['L']
            dim = simulator_args['dim']

            if np.isscalar(L):
                L = [L]*int(dim)

            else:
                L = np.array(L)


            V = np.prod(np.array(L))

            scale = V**(1./dim)

            Lk = ()

            _N = 1
            for i,_shape in enumerate(shape):
                _N *= _shape
                Lk += (_shape / L[i],) # 1 / dx


            fft_norm = np.prod(np.array(Lk))

            _dims = len(shape)
            tpl = ()
            for _d in range(_dims):
                tpl += (_d,)

            # POWERBOX IMPLEMENTATION

            mag = jax.random.normal(key1, shape=tuple(N for N in new_shape))
            
            # random phases
            pha = 2 * np.pi * jax.random.uniform(key1, shape=tuple(N for N in new_shape))
            
            # now make hermitian field (reality condition)
            revidx = (slice(None, None, -1),) * len(mag.shape)
            mag = (mag + mag[revidx]) / np.sqrt(2) 
            pha = (pha - pha[revidx]) / 2 + np.pi
            dk = mag * (np.cos(pha) + 1j * np.sin(pha)) # output is complex

            cutidx = (slice(None, -1),) * len(new_shape)

            dk = dk[cutidx]
            powers = np.concatenate((np.zeros(1), 
                        np.sqrt(P(k.flatten()[1:], A=A, B=B)))).reshape(k.shape)

            # normalize power by volume
            if simulator_args['vol_norm']:
                powers = powers/V

            fourier_field = powers * dk

            fourier_field = jax.ops.index_update(
                fourier_field,
                np.zeros(len(shape), dtype=int),
                np.zeros((1,)))

            field = np.expand_dims(np.fft.ifftn(fourier_field) * fft_norm * V, (0,))

            if simulator_args["N_scale"]:
                field *= scale

            if not simulator_args["squeeze"]:
                field = np.expand_dims(field, (0,))

            return np.array(np.real((field)), dtype='float32')

        shape = simulator_args["shape"]
        A, B = θ


        if A.shape == B.shape:
            if len(A.shape) == 0:
                return fn(rng, A, B)
            else:
                keys = jax.random.split(rng, num=A.shape[0] + 1)
                rng = keys[0]
                keys = keys[1:]
                return jax.vmap(
                    lambda key, A, B: simulator(key, (A, B), simulator_args=simulator_args)
                )(keys, A, B)
        else:
            if len(A.shape) > 0:
                keys = jax.random.split(rng, num=A.shape[0] + 1)
                rng = keys[0]
                keys = keys[1:]
                return jax.vmap(
                    lambda key, A: simulator(key, (A, B), simulator_args=simulator_args)
                )(keys, A)
            elif len(B.shape) > 0:
                keys = jax.random.split(rng, num=B.shape[0])
                return jax.vmap(
                    lambda key, B: simulator(key, (A, B), simulator_args=simulator_args)
                )(keys, B)





    def analyticFieldLikelihood(self, 
                             field_shape,
                             Δ,
                             prior,
                             k=None,
                             pk=None,
                             gridsize=20, 
                             tiling=2):

            """code for computing a gaussian field's likelihood for power spectrum parameters
                field_shape : list. shape of field input
                Δ :           array-like. FFT of the real-space field
                prior :       array-like. range over which to compute the likelihood
                k :           array-like. fourier modes over which to compute P(k)
                tiling :      list or int. tiling=2 means likelihood will be computed as 2x2 grid
                gridsize :    how large to make the likelihood surface
            """
            if k is None:
                k = self.k
            
            if pk is None:
                pk = self.pk
            
            self.field_shape = field_shape
            self.gridsize = gridsize
            if np.isscalar(tiling):
                self.tiling = [tiling]*2
            else:
                self.tiling = tiling    
            #self.tilesize = gridsize // tiling
            self.N = np.sqrt(np.prod(np.array(field_shape)))  # should just be N for NxN grid
            self.prior = prior

            self.k = k
            self.Δ = Δ

        def Pk(self, k, A=1, B=0.5):
            return self.pk(k, A, B)


            return np.diag(pk)
        def log_likelihood(self, k, A, B, Δ):
            Δ = Δ.flatten()
            k = k

            dlength = len(k.flatten())

            def fn(_A, _B):

                nrm = np.pad(np.ones(dlength-2)*2, (1,1), constant_values=1.)
                nrm = jax.ops.index_update(
                  nrm, np.array([[0],[(dlength-2)]]), np.array([[1],[1]]))
                #nrm = 1

                powers = np.concatenate((np.ones(1), 
                        (self.Pk(k.flatten()[1:], A=_A, B=_B))))

                # covariance is P(k)
                C = powers * nrm
                invC = np.concatenate((np.zeros(1), 
                   (1./self.Pk(k.flatten()[1:], A=_A, B=_B))))

                logdetC = np.sum(np.log(C))
                pi2 = np.pi * 2.
                m_half_size = -0.5 * len(Δ)
                exponent = - 0.5 * np.sum(np.conj(Δ) * invC * Δ)
                norm = -0.5 * logdetC + m_half_size*np.log(pi2) 
                return (exponent + norm)
            return jax.vmap(fn)(A, B)

        def get_likelihood(self, return_grid=False, shift=None):
            A_start = self.prior[0][0]
            A_end = self.prior[1][0]
            B_start = self.prior[0][1]
            B_end = self.prior[1][1]

            region_size = [self.gridsize // self.tiling[i] for i in range(len(self.tiling))]

            print("computing likelihood on a %dx%d grid \n \
            in tiles of size %dx%d"%(self.gridsize, self.gridsize, region_size[0], region_size[1]))


            def get_like_region(A0, A1, B0, B1, qsize):
                A_range = np.linspace(A0, A1, qsize)
                B_range = np.linspace(B0, B1, qsize)
                A, B = np.meshgrid(A_range, B_range)

                return (self.log_likelihood(k,
                    A.ravel(), B.ravel(), Δ).reshape(qsize,qsize))



            A_incr = (A_end - A_start) / self.tiling[0]
            B_incr = (B_end - B_start) / self.tiling[1]

            # marks the ends of linspace
            A_starts = [A_start + (i)*A_incr for i in range(self.tiling[0])]
            A_ends = [A_start + (i+1)*A_incr for i in range(self.tiling[0])]
            B_starts = [B_start + (i)*B_incr for i in range(self.tiling[1])]
            B_ends = [B_start + (i+1)*B_incr for i in range(self.tiling[1])]

            _like_cols = []
            for _col in range(self.tiling[0]):
                # slide horizontally in A
                _like_row = []
                for _row in range(self.tiling[1]):
                    # slide vertically in B

                    _like = get_like_region(A_starts[_row], 
                                            A_ends[_row], 
                                            B_starts[_col], 
                                            B_ends[_col],
                                            region_size[0],
                                            )
                    _like_row.append(_like)

                _like_cols.append(np.concatenate(_like_row, axis=1))

            _log_likelihood = np.real(np.concatenate(_like_cols, axis=0))

            if shift is None:
                shift = np.max(_log_likelihood)

            print('shift', shift)
            print('loglike mean', np.mean(_log_likelihood))

            _log_likelihood = _log_likelihood - shift

            if return_grid:
                _A_range = np.linspace(self.prior[0,0], self.prior[1,0], self.gridsize)
                _B_range = np.linspace(self.prior[0,0], self.prior[1,0], self.gridsize)
                return np.exp(_log_likelihood), _A_range, _B_range

            return np.exp(_log_likelihood)



        def plot_contours(self, ax=None, 
                          θ_ref=None, shift=None, 
                          xlabel='A', ylabel='B', 
                          return_like=True):

            _like, _A, _B = self.get_likelihood(return_grid=True, shift=shift)


            _A, _B = np.meshgrid(_A, _B)

            if ax is None:
                fig,ax = plt.subplots(figsize=(10,10))

            mesh = ax.contourf(_A, _B, _like)
            plt.colorbar(mesh, ax=ax)

            if θ_ref is not None:
                ax.scatter(θ_ref[0], θ_ref[1], zorder=10, marker='+', s=100, color='r')

            ax.set_xlabel('A')
            ax.set_ylabel('B')

            if return_like:
                return _like, ax

            else:
                return ax

        def plot_corner(self, ax=None, label="Analytic likelihood"):

            _like, _A_range, _B_range = self.get_likelihood(return_grid=True)

            likelihoodA = _like.sum(0)
            likelihoodA /= likelihoodA.sum() * (_A_range[1] -  _A_range[0])
            likelihoodB = _like.sum(1)
            likelihoodB /= likelihoodB.sum() * (_B_range[1] -  _B_range[0])
            sorted_marginal = np.sort(_like.flatten())[::-1]
            cdf = np.cumsum(sorted_marginal / sorted_marginal.sum())
            value = []
            for level in [0.95, 0.68]:
                this_value = sorted_marginal[np.argmin(np.abs(cdf - level))]
                if len(value) == 0:
                    value.append(this_value)
                elif this_value <= value[-1]:
                    break
                else:
                    value.append(this_value)


            # add in the likelihood estimate
            ax[0, 0].plot(_A_range, likelihoodA, color="C2", label=label)
            ax[0, 1].axis("off")
            ax[1, 0].contour(_A_range, _B_range, _like, levels=value, colors="C2")
            ax[1, 1].plot(likelihoodB, _B_range, color="C2", label=label)

            return ax
