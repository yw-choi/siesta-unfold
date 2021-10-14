import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sisl


def main():

    # Plotting options
    figsize = (3, 3)
    dpi = 200
    emin, emax = -2, 2
    output = 'bands.png'

    title = os.path.basename(os.getcwd())
    siesta_input = 'input.fdf'
    siesta_outfn = 'siesta.stdout'

    fdf = sisl.get_sile(siesta_input)
    label = fdf.get('SystemLabel')
    xvfile = f'{label}.XV'
    bandsfile = f'{label}.bands'

    print('[SIESTA Band Structure]')
    print(f'dirname = {title}')
    print(f'input  = {siesta_input}')
    print(f'XV file = {xvfile}')
    print(f'bands file = {bandsfile}')
    print(f'output file = {siesta_outfn}')

    out = sisl.io.siesta.outSileSiesta(siesta_outfn)
    if not out.completed():
        print('Error: SIESTA run is not completed')
        return

    geom = sisl.get_sile(xvfile).read_geometry()
    kticks, kpoints, bands = sisl.get_sile(bandsfile).read_data()
    nk, nspin, nbnd = bands.shape

    # Band Gap Information
    qtot = read_qtot(siesta_outfn)
    ivbm = int(qtot/2)-1
    icbm = int(qtot/2)
    vbm = []
    cbm = []
    bandgaps = []
    for ispin in range(nspin):
        ik_vbm = np.argmax(bands[:, ispin, ivbm])
        ik_cbm = np.argmin(bands[:, ispin, ivbm])
        vbm.append((ik_vbm, bands[ik_vbm, ispin, ivbm]))
        cbm.append((ik_cbm, bands[ik_cbm, ispin, icbm]))
        bandgaps.append(bands[ik_vbm, ispin, icbm]-bands[ik_vbm, ispin, ivbm])

    # Print band gap info
    print('# Band Gap Information')
    for ispin in range(nspin):
        print(f'ispin = {ispin}')
        print(
            f'ik_vbm, e_vbm (eV) = {ik_vbm}, {bands[ik_vbm,ispin,ivbm]:16.8f}')
        print(
            f'ik_cbm, e_cbm (eV) = {ik_cbm}, {bands[ik_cbm,ispin,icbm]:16.8f}')
        print(f'band gap (eV) = {bandgaps[ispin]:12.3f}')

    # Band Structure Plotting
    print('# Band Structure Plotting')
    plt.figure(figsize=figsize)
    colors = ['k', 'b']
    # for ispin in range(nspin):
    #     plt.plot(kpoints, bands[:, ispin, 400:500],
    #              ls='-', lw=.1, c=colors[ispin])

    # unfolding weights
    enk = np.loadtxt('enk.1.txt')
    wnk = np.loadtxt('wnk.1.txt')
    nk, nbnd = enk.shape
    vmin, vmax = 0, wnk.max()
    for ib in range(nbnd):
        # plt.plot(np.arange(unfold.nk), enk[0, :, ib], 'k-', lw=.5)
        # c = np.zeros((nk, 4))
        # c[:, 0] = 1
        # c[:, 3] = wnk[:, ib]/vmax
        # plt.scatter(np.arange(nk), enk[:, ib], c='k', s=1)
        plt.scatter(kpoints, enk[:, ib], c='red',
                    edgecolor='none', s=(wnk[:, ib]/vmax)*10)

    # Plot styling
    gridlinespec = {'color': 'darkgrey',
                    'linestyle': ':',
                    'linewidth': 0.5}
    plt.grid(True, axis='x', **gridlinespec)
    plt.axhline(0,  **gridlinespec)
    plt.title(title)
    plt.ylabel('Energy (eV)')
    plt.xticks(kticks[0], kticks[1])
    plt.xlim(kticks[0][0], kticks[0][-1])
    # plt.ylim(emin, emax)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    print(f'saved plot to {output}')

    return


def read_qtot(fn):
    qtot = None
    with open(fn, 'r') as f:
        for l in f:
            if 'Qtot' in l:
                qtot = float(l.split()[-1])
                break
    if qtot is None:
        raise Exception(f'Qtot not found in {fn}')
    return qtot


main()
