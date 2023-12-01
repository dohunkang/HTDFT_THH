import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rc('font', family='Arial')

"""
author: @dohunkang
A python code to summarize the overall HT-DFT calculation on host-guest nanoparticle surfaces
"""

energies = pickle.load(open("../data/energies_w_stable.pickle", "rb"))
result_1ML = np.loadtxt("../data/result_1ML.txt", skiprows=1, dtype={'names': ('host', 'guest', 'surf', 'Etot', 'nhost', 'nsub', 'surfA', 'type'),
                     'formats': ('U2', 'U2', 'U3', 'f8', 'i4', 'i4', 'f8', '<U16')})
result_hqML = np.loadtxt("../data/result_hqML.txt", skiprows=1, dtype={'names': ('host', 'guest', 'surf', 'Etot', 'nhost', 'nsub', 'latt_a', 'latt_b', 'surfA', 'type'),
                     'formats': ('U2', 'U2', 'U3', 'f8', 'i4', 'i4', 'f8', 'f8', 'f8','<U16')})
host = ['Co', 'Ni', 'Cu', 'Rh', 'Pd', 'Ag', 'Ir', 'Pt', 'Au']
guest  = ['Zn', 'Ga', 'Ge', 'As', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Hg', 'Tl', 'Pb', 'Bi']

result_dict = {}
for hh in host:
    for gg in guest:
        for surf in ['100', '110', '111', '210']:
            result_dict[hh + "-" + gg + "_" + surf] = 10 * np.ones((4, 100))

for res in result_1ML:
    energy = energies[res['host'] + "-" + res['guest']]
    mu_host = energy[0]
    mu_guest = energy[1]
    mu_alloy = energy[2]
    ii = host.index(res['host'])
    idx = guest.index(res['guest'])
    surfE1 = (res['Etot'] - res['nhost'] * mu_host - res['nsub'] * mu_guest)/2/res['surfA']*1.60218*10
    surfE2 = (res['Etot'] - res['nhost'] * mu_host - res['nsub'] * mu_alloy)/2/res['surfA']*1.60218*10
    
    hh = res['host']
    gg = res['guest']
    surf = res['surf']
    for ii in range(4):
        if result_dict[hh + "-" + gg + "_" + surf][ii][0] == 10:
            result_dict[hh + "-" + gg + "_" + surf][ii] = np.linspace(surfE1, surfE2, 100)
            break

for res in result_hqML:
    energy = energies[res['host'] + "-" + res['guest']]
    mu_host = energy[0]
    mu_guest = energy[1]
    mu_alloy = energy[2]
    ii = host.index(res['host'])
    idx = guest.index(res['guest'])
    surfE1 = (res['Etot'] - res['nhost'] * mu_host - res['nsub'] * mu_guest)/2/res['surfA']*1.60218*10
    surfE2 = (res['Etot'] - res['nhost'] * mu_host - res['nsub'] * mu_alloy)/2/res['surfA']*1.60218*10
    
    hh = res['host']
    gg = res['guest']
    surf = res['surf']
    for ii in range(4):
        if result_dict[hh + "-" + gg + "_" + surf][ii][0] == 10:
            result_dict[hh + "-" + gg + "_" + surf][ii] = np.linspace(surfE1, surfE2, 100)
            break

fig, ax = plt.subplots(len(host), len(guest), figsize=(45,31))
for ii in range(len(host)):
    for jj in range(len(guest)):
        ax[ii,jj].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax[ii,jj].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
for ii in range(len(host)):
    ax[ii, 0].set_ylabel(host[ii],fontsize=20)
for idx in range(len(guest)):
    ax[0, idx].set_title(guest[idx], fontsize=20)

for ii, hh in enumerate(host):
    for jj, gg in enumerate(guest):
        energy = energies[hh + "-" + gg]
        mu_guest = energy[1]
        mu_alloy = energy[2]
        if mu_guest == mu_alloy:
            point100, = ax[ii, jj].plot(mu_guest, np.min(result_dict[hh + "-" + gg + "_100"][:,0]), 'o', color='orange')
            point110, = ax[ii, jj].plot(mu_guest, np.min(result_dict[hh + "-" + gg + "_110"][:,0]), '^', color='green')
            point111, = ax[ii, jj].plot(mu_guest, np.min(result_dict[hh + "-" + gg + "_111"][:,0]), '*', color='blue')
            point210, = ax[ii, jj].plot(mu_guest, np.min(result_dict[hh + "-" + gg + "_210"][:,0]), 's', color='red')
        else:
            line100, = ax[ii, jj].plot(np.linspace(mu_guest, mu_alloy, 100), np.min(result_dict[hh + "-" + gg + "_100"], axis=0), '--', color='orange', linewidth="3")
            line110, = ax[ii, jj].plot(np.linspace(mu_guest, mu_alloy, 100), np.min(result_dict[hh + "-" + gg + "_110"], axis=0), ':', color='green', linewidth="3")
            line111, = ax[ii, jj].plot(np.linspace(mu_guest, mu_alloy, 100), np.min(result_dict[hh + "-" + gg + "_111"], axis=0), '-.', color='blue', linewidth="3")
            line210, = ax[ii, jj].plot(np.linspace(mu_guest, mu_alloy, 100), np.min(result_dict[hh + "-" + gg + "_210"], axis=0), '-', color='red', linewidth="3")

    
lgd = fig.legend([line100, point100, line110, point110, line111, point111, line210, point210], 
           ['(100)', '(100)', '(110)', '(110)', '(111)', '(111)', '(210)', '(210)'], 
           loc = 'lower center', ncol=4, bbox_to_anchor=(0.5,-0.03), fontsize=20, edgecolor="black")
lgd.get_frame().set_alpha(0)
lgd.get_frame().set_facecolor((0, 0, 0, 0))
fig.tight_layout(pad=1.5)
plt.savefig("overall_public.png", bbox_extra_artists=(lgd,), bbox_inches='tight', transparent=True)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


# Cited from https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)

    return newcmap
	
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
htmap_pure = np.zeros((len(host), len(guest)))
htmap_alloy = np.zeros((len(host), len(guest)))
htmap_pref = np.zeros((len(host), len(guest)))

for ii, hh in enumerate(host):
    for jj, gg in enumerate(guest):
        energy = energies[hh + "-" + gg]
        mu_guest = energy[1]
        mu_alloy = energy[2]
        if mu_guest == mu_alloy:
            point100 = np.min(result_dict[hh + "-" + gg + "_100"][:,0])
            point110 = np.min(result_dict[hh + "-" + gg + "_110"][:,0])
            point111 = np.min(result_dict[hh + "-" + gg + "_111"][:,0])
            point210 = np.min(result_dict[hh + "-" + gg + "_210"][:,0])
            htmap_pure[ii, jj] = point210 - min((point100, point110, point111))
            htmap_alloy[ii, jj] = point210 - min((point100, point110, point111))
            surfE = np.zeros(4)
            surfE[0] = np.min(result_dict[hh + "-" + gg + "_100"][:,0])
            surfE[1] = np.min(result_dict[hh + "-" + gg + "_110"][:,0])
            surfE[2] = np.min(result_dict[hh + "-" + gg + "_111"][:,0])
            surfE[3] = np.min(result_dict[hh + "-" + gg + "_210"][:,0])
            preference = 1 if np.argmin(surfE) == 3 else 0
            htmap_pref[ii,jj] = preference
        else:
            line100 = np.min(result_dict[hh + "-" + gg + "_100"], axis=0)
            line110 = np.min(result_dict[hh + "-" + gg + "_110"], axis=0)
            line111 = np.min(result_dict[hh + "-" + gg + "_111"], axis=0)
            line210 = np.min(result_dict[hh + "-" + gg + "_210"], axis=0)
            htmap_pure[ii, jj] = line210[0] - min((line100[0], line110[0], line111[0]))
            htmap_alloy[ii, jj] = line210[-1] - min((line100[-1], line110[-1], line111[-1]))
            surfE = np.zeros((4, 100))
            surfE[0] = np.min(result_dict[hh + "-" + gg + "_100"], axis=0)
            surfE[1] = np.min(result_dict[hh + "-" + gg + "_110"], axis=0)
            surfE[2] = np.min(result_dict[hh + "-" + gg + "_111"], axis=0)
            surfE[3] = np.min(result_dict[hh + "-" + gg + "_210"], axis=0)
            preference = np.average(np.argmin(surfE, axis=0) == 3)
            htmap_pref[ii,jj] = preference

# sns.set(font_scale=1.0)
plt.figure()
ax = sns.heatmap(htmap_pure, yticklabels=host, xticklabels=guest, square=True, cmap=matplotlib.cm.bwr, vmin=-1.5, vmax=1.5, cbar_kws={"shrink":0.75, 'label': 'J/m$^2$'}, clip_on=False, linewidths=0.01, linecolor='white')
ax.set(xlabel="Guest", ylabel="Host", title="E(210) - min(E(100), E(110), E(111)) pure")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

plt.figure()
ax = sns.heatmap(htmap_alloy, yticklabels=host, xticklabels=guest, square=True, cmap=matplotlib.cm.bwr, vmin=-1.5, vmax=1.5, cbar_kws={"shrink":0.75, 'label': 'J/m$^2$'}, clip_on=False, linewidths=0.01, linecolor='white')
ax.set(xlabel="Guest", ylabel="Host", title="E(210) - min(E(100), E(110), E(111)) alloy")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 

plt.figure()
ax = sns.heatmap(htmap_pref, yticklabels=host, xticklabels=guest, cmap=shiftedColorMap(matplotlib.cm.bwr.reversed(), 0, 0.6, 1), square=True, vmin=0, vmax=1, cbar_kws={"shrink":0.75}, clip_on=False, linewidths=0.01, linecolor='white')
ax.set(xlabel="Guest", ylabel="Host", title="Preference")
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top') 