# %%
import time
import sys
import numpy as np
import os
import sisl
from datetime import datetime as dt
import logging
import matplotlib.pyplot as plt
import itertools

try:
    from mpi4py import MPI
except:
    MPI = None


def main():

    ################################################
    # run parameters
    ################################################
    fdf_file = f'./input.fdf'
    ecut = 500  # eV
    a = 2.471153
    c = 25
    cell_uc = np.array([
        [np.sqrt(3)/2*a, -1/2*a, 0],
        [np.sqrt(3)/2*a, +1/2*a, 0],
        [0,      0, c]
    ])
    lowest_band = 400
    highest_band = 500

    ################################################
    # MPI Setup
    ################################################
    if MPI:
        comm = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        nprocs = 1
        rank = 0

    ################################################
    # Start
    ################################################
    if rank == 0:
        print(f'Unfolding band structure from SIESTA')
        print(f'Young Woo Choi, 2021/07/11')
        print(f'Parameters')

    unfold = Unfold(fdf_file,
                    ecut=ecut,
                    cell_uc=cell_uc,
                    lowest_band=lowest_band,
                    highest_band=highest_band,
                    mpi_comm=comm)
    enk, wnk = unfold.calc_weights()
    if rank == 0:
        unfold.timer.print_clocks()
        for ispin in range(unfold.nspin):
            np.savetxt(f'enk.{ispin+1}.txt', enk[ispin, :, :])
            np.savetxt(f'wnk.{ispin+1}.txt', wnk[ispin, :, :])

    return


class Unfold:
    def __init__(self, fdfname, cell_uc, ecut,
                 lowest_band=None, highest_band=None, mpi_comm=None):

        self.timer = Timer()
        self.timer.start_clock('unfold')

        self.timer.start_clock('init')
        self.fdfname = fdfname
        self.ecut = ecut
        self.lowest_band = lowest_band
        self.highest_band = highest_band
        self.mpi_comm = mpi_comm
        if mpi_comm is not None:
            self.mpi_nprocs = mpi_comm.Get_size()
            self.mpi_rank = mpi_comm.Get_rank()
        else:
            self.mpi_nprocs = 1
            self.mpi_rank = 0

        if self.mpi_rank == 0:
            self.print_time(f'[Unfold init]')

        if not os.path.exists(fdfname):
            self.error(f'{fdfname} does not exist.')
            return

        self.fdf = sisl.io.siesta.fdfSileSiesta(fdfname)
        self.geom = self.fdf.read_geometry()
        self.cell = self.geom.cell
        self.bcell = 2*np.pi*np.linalg.inv(self.cell).T
        self.cell_uc = cell_uc
        self.bcell_uc = 2*np.pi*np.linalg.inv(self.cell_uc).T
        self.fdf_path = self.fdf.dir_file()
        self.outdir = self.fdf_path.parents[0]
        self.system_label = self.fdf.get('SystemLabel')

        self.ef = self.fdf.read_fermi_level()

        self._siesta = sisl.io.siesta._siesta

        # setup wfsx
        self.wfsx_file = f'{self.outdir}/{self.system_label}.selected.WFSX'
        if not os.path.exists(self.wfsx_file):
            self.error('[Error] .WFSX file not found')
            return

        # First query information
        self.nspin, self.nou, self.nk, self.Gamma = self._siesta.read_wfsx_sizes(
            self.wfsx_file)
        if self.nspin in [4, 8]:
            self.nspin = 1  # only 1 spin
            self._read_wfsx = self._siesta.read_wfsx_index_4
        elif self.Gamma:
            self._read_wfsx = self._siesta.read_wfsx_index_1
        else:
            self._read_wfsx = self._siesta.read_wfsx_index_2

        # setup band range
        if self.lowest_band is None:
            self.lowest_band = 1

        if self.highest_band is None:
            self.highest_band = self.nou

        self.nbnd = self.highest_band-self.lowest_band+1

        self.basis = self.fdf.read_basis()

        self.igvec_uc = self.find_igvec(self.bcell_uc, self.ecut)
        self.gvec_uc = self.igvec_uc.dot(self.bcell_uc)

        self.gsplit = np.array_split(self.gvec_uc, self.mpi_nprocs)

        self.timer.stop_clock('init')
        if self.mpi_rank == 0:
            print(f'len(self.gvec_uc)={len(self.gvec_uc)}')
            for i in range(self.mpi_nprocs):
                ng_loc = len(self.gsplit[i])
                print(f'rank={i}, ng_loc={ng_loc}')
        return

    def calc_weights(self):
        self.timer.start_clock('calc_weights')

        enk = np.zeros((self.nspin, self.nk, self.nbnd))
        wnk = np.zeros((self.nspin, self.nk, self.nbnd))

        Ang2Bohr = sisl.unit.unit_convert('Ang', 'Bohr')

        pos = self.geom.axyz()
        io2ia = np.zeros(self.nou, dtype=np.int32)
        io2isp = np.zeros(self.nou, dtype=np.int32)
        io2ioa = np.zeros(self.nou, dtype=np.int32)
        for ia, io in self.geom.iter_orbitals(local=False):
            io2ia[io] = ia
            io2ioa[io] = io-self.geom.atoms.firsto[ia]
            io2isp[io] = self.geom.atoms.specie[ia]

        phase = np.zeros(
            (len(self.gsplit[self.mpi_rank]), len(pos)), dtype=np.complex128)
        for ig, g in enumerate(self.gsplit[self.mpi_rank]):
            phase[ig, :] = np.exp(-1j*pos.dot(g))

        for ispin, ik in itertools.product(range(1, self.nspin+1), range(1, self.nk+1)):
            if self.mpi_rank == 0:
                self.print_time(f'ik={ik}')
            self.timer.start_clock('ik')

            # k is in Bohr^{-1} unit
            self.timer.start_clock('read_wfsx')
            if self.mpi_comm:
                if self.mpi_rank == 0:
                    k_bohr, _, nwf = self._siesta.read_wfsx_index_info(
                        self.wfsx_file, ispin, ik)
                    idx, eig, state = self._read_wfsx(
                        self.wfsx_file, ispin, ik, self.nou, nwf)
                else:
                    k_bohr = None
                    nwf = None
                    idx = None
                    eig = None
                    state = None
                k_bohr = self.mpi_comm.bcast(k_bohr, 0)
                nwf = self.mpi_comm.bcast(nwf, 0)
                idx = self.mpi_comm.bcast(idx, 0)
                eig = self.mpi_comm.bcast(eig, 0)
                state = self.mpi_comm.bcast(state, 0)
            else:
                k_bohr, _, nwf = self._siesta.read_wfsx_index_info(
                    self.wfsx_file, ispin, ik)
                idx, eig, state = self._read_wfsx(
                    self.wfsx_file, ispin, ik, self.nou, nwf)

            self.timer.stop_clock('read_wfsx')

            if self.highest_band > nwf or self.lowest_band > nwf:
                print(f'[Error] nwf < highest_band')
                exit(-1)
            ib0 = self.lowest_band-1
            ib1 = ib0+self.nbnd

            enk[ispin-1, ik-1, :] = eig[ib0:ib1]-self.ef

            self.timer.start_clock(f'matel')
            matel = np.zeros(self.nbnd, dtype=np.complex128)
            k = k_bohr * Ang2Bohr

            for ig, g in enumerate(self.gsplit[self.mpi_rank]):
                kpg = k+g
                self.timer.start_clock(f'orbfac')
                PAO_FT = self.calc_PAO_FT(kpg)
                ftfac = np.zeros(self.nou, dtype=np.complex128)

                for ia, io in self.geom.iter_orbitals(local=False):
                    # ftfac[io] = PAO_FT[io2isp[io]][io2ioa[io]]
                    if ia >= 881:
                        ftfac[io] = PAO_FT[io2isp[io]][io2ioa[io]]
                    else:
                        ftfac[io] = PAO_FT[io2isp[io]][io2ioa[io]]*0.1
                orbfac = phase[ig, io2ia[:]] * ftfac
                self.timer.stop_clock(f'orbfac')

                self.timer.start_clock(f'matel_sum')
                matel = np.einsum('o,on->n', orbfac, state[:, ib0:ib1])
                matel /= self.geom.volume
                wnk[ispin-1, ik-1, :] += np.abs(matel)**2
                self.timer.stop_clock(f'matel_sum')
            self.timer.stop_clock(f'matel')

            self.timer.stop_clock('ik')

        if self.mpi_nprocs > 1:
            wnk = self.mpi_comm.allreduce(sendobj=wnk, op=MPI.SUM)

        return enk, wnk

    def calc_PAO_FT(self, k, nr=300):
        """
        ------
        input
        ------
        k(3) k-vector (kx,ky,kz)
        ------
        output
        ------
        PAOFT integral exp(-ivec{k}*vec{r}) PAO(vec{r}) dV
        """
        import scipy.special as sp
        from scipy.integrate import simps

        knorm = np.linalg.norm(k)
        # gamma point
        if knorm < 1e-12:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(k[2]/knorm)
            phi = np.arctan2(k[1], k[0])

        PAOFT = []
        for isp, specie in enumerate(self.basis):
            PAOFT.append([])

            orbs = []
            for io, orb in enumerate(specie.orbitals):
                m = orb.m
                l = orb.l

                rgrid = np.linspace(0, orb.R, nr)
                phi_r = orb.radial(rgrid, is_radius=True)
                kr = knorm * rgrid
                j_l = sp.spherical_jn(l, kr)
                sbt = simps(rgrid*rgrid*j_l*phi_r, rgrid)

                FT = 4.0*np.pi*(-1)**m*(-1j)**l*sbt
                if m < 0:
                    # note that scipy uses different names for theta, phi
                    FT *= 1.j/np.sqrt(2.0)*(sp.sph_harm(m, l, phi,
                                                        theta)-(-1)**m*sp.sph_harm(-m, l, phi, theta))
                elif m == 0:
                    FT *= sp.sph_harm(0, l, phi, theta)
                else:  # m>0
                    FT *= 1.0/np.sqrt(2.0)*(sp.sph_harm(-m, l, phi,
                                                        theta)+(-1)**m*sp.sph_harm(m, l, phi, theta))

                PAOFT[isp].append(FT)

        return PAOFT

    def find_igvec(self, bcell_in, ecut_in):
        """Find G(n1,n2,n3) vectors such that {(n1,n2,n3).dot(bcell)}^2 < ecut

        Args:
            bcell_in (float64, 3x3): reciprocal lattice vectors (Ang^{-1})
            ecut_in (float64): enery cutoff in eV

        Returns:
            int32, array: list of integers (n1,n2,n3)
        """
        eV2Ry = sisl.unit.unit_convert('eV', 'Ry')
        Ang2Bohr = sisl.unit.unit_convert('Ang', 'Bohr')

        bcell = bcell_in / Ang2Bohr
        ecut = ecut_in * eV2Ry

        # find nmax such that (nmax*bcell[idir,:])^2 < ecut along each direction.
        b2 = np.sum(bcell**2, axis=1)
        nmax = np.int32(2*np.sqrt(ecut/b2))

        n1 = np.arange(-nmax[0], nmax[0]+1)
        n2 = np.arange(-nmax[1], nmax[1]+1)
        n3 = np.arange(-nmax[2], nmax[2]+1)
        # n1grid, n1grid, n3grid = np.meshgrid(n1, n2, n3)
        igvec_all = np.int32([(i, j, k) for i in n1 for j in n2 for k in n3])
        gvec = igvec_all.dot(bcell)
        g2 = np.einsum('ix,ix->i', gvec, gvec)

        igvec = igvec_all[np.where(g2 < ecut)]
        # test = igvec.dot(bcell_uc)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(test[:, 0], test[:, 1], test[:, 2], s=1, c='r')
        # c = plt.Circle((0, 0), np.sqrt(ecut), color='k', fill=False)
        # ax.add_patch(c)
        # plt.savefig('test.png')

        return igvec

    def error(self, msg):
        print(msg)
        exit(-1)
        return

    def print_time(self, msg):
        t = dt.now()
        print(f'{msg} @ {t}')
        sys.stdout.flush()


class Timer:

    def __init__(self):
        self.clocks = {}

    def start_clock(self, key):
        if key not in self.clocks:
            self.clocks[key] = {
                'is_on': False,
                'start_time': None,
                'total_time': 0.0,
                'counter': 0,
            }

        if self.clocks[key]['is_on']:
            print(f'Clock is already on. {key}')
            exit(-1)

        self.clocks[key]['is_on'] = True
        self.clocks[key]['start_time'] = time.time()
        self.clocks[key]['counter'] += 1
        return

    def stop_clock(self, key):
        if key not in self.clocks or self.clocks[key]['is_on'] == False:
            print(f'cannot stop clock {key}')
            exit(-1)
        self.clocks[key]['is_on'] = False
        self.clocks[key]['total_time'] = time.time() - \
            self.clocks[key]['start_time']
        self.clocks[key]['start_time'] = None
        return

    def print_clocks(self):
        for k, v in self.clocks.items():
            print(f'{k:25s} {v["total_time"]:.3f} {v["counter"]}')


if __name__ == "__main__":
    main()
