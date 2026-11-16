"""Reproduce the two-observation DUET lecture demo from two AISHELL-1 WAVs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib import font_manager
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal import istft, stft

FS = 16000
NFFT = 512
HOP = 128
GUARD = 128
MIN_F = 125
MAX_F = 1400
ATTENUATION_EDGES = np.linspace(-10, 10, 81)
DELAY_EDGES = np.linspace(-7, 7, 81)
CASES = [
    ("distinct", "易 · 两个峰分开", (-3, 3), (0.76, 1.30)),
    ("nearby", "中 · 两个峰靠近", (-1, 1), (1.0, 1.0)),
    ("degenerate", "难 · 两个峰重合", (0, 0), (1.0, 1.0)),
]
STFT_ARGS = dict(fs=FS, window="hann", nperseg=NFFT, noverlap=NFFT-HOP)


def read_speech(first, second):
    source = []
    for path in (first, second):
        rate, pcm = wavfile.read(path)
        if rate != FS or pcm.ndim != 1 or pcm.dtype != np.int16:
            raise ValueError(f"{path} must be 16 kHz mono PCM16")
        source.append(pcm.astype(np.float64) / 32768)
    length = min(map(len, source))
    clips = np.stack([x[:length] for x in source])
    clips -= np.mean(clips, axis=1, keepdims=True)
    clips /= np.sqrt(np.mean(clips**2, axis=1, keepdims=True))
    return np.pad(clips*0.065, ((0,0),(GUARD,GUARD)))


def shift(signal, samples):
    shifted = np.zeros_like(signal)
    if samples > 0:
        shifted[samples:] = signal[:-samples]
    elif samples < 0:
        shifted[:samples] = signal[-samples:]
    else:
        shifted[:] = signal
    return shifted


def build_observations(sources, delays, gains):
    mix = np.sum(sources, axis=0)
    second = sum(g*shift(s,d) for s,d,g in zip(sources,delays,gains,strict=True))
    return mix, second


def estimate_peaks(frequencies, first, second):
    """Use reliable low-frequency samples to avoid wrapping the phase ratio."""
    a1, a2 = np.abs(first), np.abs(second)
    active = (a1 > a1.max()*1e-3) & (a2 > a2.max()*1e-3)
    selected = active & (frequencies[:,None]>=MIN_F) & (frequencies[:,None]<=MAX_F)
    bins, frames = np.nonzero(selected)
    ratio = second[bins,frames] / first[bins,frames]
    gain_db = 20*np.log10(np.abs(ratio))
    delay = -np.angle(ratio)*FS/(2*np.pi*frequencies[bins])
    weights = a1[bins,frames]*a2[bins,frames]
    raw = np.histogram2d(gain_db, delay, (ATTENUATION_EDGES,DELAY_EDGES),weights=weights)[0]
    density = gaussian_filter(raw, sigma=1.1)
    maxima = (density == maximum_filter(density, size=(7,9))) & (density > 0.10*density.max())
    candidates = sorted(np.argwhere(maxima), key=lambda indices: density[tuple(indices)], reverse=True)
    peaks = []
    for row,col in candidates[:2]:
        a = (ATTENUATION_EDGES[row]+ATTENUATION_EDGES[row+1])/2
        d = (DELAY_EDGES[col]+DELAY_EDGES[col+1])/2
        points = (np.abs(gain_db-a)<0.5)&(np.abs(delay-d)<0.35)
        if np.any(points):
            a = np.average(gain_db[points],weights=weights[points])
            d = np.average(delay[points],weights=weights[points])
        peaks.append((float(a),float(d)))
    return density, sorted(peaks,key=lambda peak: peak[1])


def binary_mask_reconstruct(frequencies, first, second, peaks, length):
    if len(peaks)!=2:
        return []
    costs=[]
    for attenuation, delay in peaks:
        gain = 10**(attenuation/20)
        steering = gain*np.exp(-2j*np.pi*frequencies*delay/FS)
        costs.append(np.abs(second-steering[:,None]*first)**2/(1+gain**2))
    bins=np.argmin(costs,axis=0)
    return [istft(first*(bins==index),**STFT_ARGS)[1][:length] for index in range(2)]


def si_sdr(estimate, target):
    estimate=estimate-estimate.mean()
    target=target-target.mean()
    projection=target*np.dot(estimate,target)/np.dot(target,target)
    return float(10*np.log10(max(np.dot(projection,projection),1e-20)/
                            max(np.linalg.norm(estimate-projection)**2,1e-20)))


def display_audio(path, signal, common_rms=0.065):
    crop=signal[GUARD:-GUARD]
    scale=min(common_rms/max(np.sqrt(np.mean(crop**2)),1e-10),0.89/max(np.max(np.abs(crop)),1e-10))
    wavfile.write(path,FS,np.round(np.clip(crop*scale,-0.99,0.99)*32767).astype(np.int16))


def plot_clusters(densities, rows, out):
    cjk=Path("/System/Library/Fonts/Supplemental/Songti.ttc")
    if cjk.exists():
        font_manager.fontManager.addfont(cjk)
        plt.rcParams["font.family"]="Songti SC"
    plt.rcParams["axes.unicode_minus"]=False
    cmap=LinearSegmentedColormap.from_list("source_activity",["#fff","#f6d5cd","#bc4c59","#861a35"])
    def draw(ax,density,row):
        ax.pcolormesh(DELAY_EDGES,ATTENUATION_EDGES,density,shading="flat",cmap=cmap,
                      norm=PowerNorm(gamma=.45,vmin=0,vmax=density.max()),rasterized=True)
        for a,d in row["peaks"]:
            ax.scatter(d,a,marker="+",s=96,linewidths=1.55,color="#202526")
        ax.set(xlim=(-4,4),ylim=(-4,4),xticks=(-4,-2,0,2,4),yticks=(-4,-2,0,2,4))
        ax.set_xlabel("相对时延（采样点）",fontsize=11,labelpad=5)
        ax.set_ylabel("相对幅度（dB）",fontsize=11,labelpad=5)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color("#d6c7c5")
        ax.tick_params(axis="both",colors="#49413f",labelsize=10,length=3)
        ax.set_title(row["title"],loc="left",fontsize=15,pad=10,color="#812234")
    for density,row in zip(densities,rows,strict=True):
        fig,ax=plt.subplots(figsize=(5.4,3.35),layout="constrained")
        draw(ax,density,row)
        fig.savefig(out/f"{row['id']}-hist.png",dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.0,3.25),layout="constrained")
    for ax,density,row in zip(axes,densities,rows,strict=True):
        draw(ax,density,row)
    fig.savefig(out/"duet-clusters.png",dpi=210)
    plt.close(fig)


def run(first, second, out):
    """Prepare histograms, exact settings, measurements and clickable WAV tracks."""
    out.mkdir(parents=True,exist_ok=True)
    sources=read_speech(first,second)
    base_mix = sources.sum(axis=0)
    baseline = [si_sdr(base_mix[GUARD:-GUARD],s[GUARD:-GUARD]) for s in sources]
    rows,densities = [],[]
    display_audio(out/"mixture.wav",base_mix)
    for ident,title,delays,gains in CASES:
        x1,x2 = build_observations(sources,delays,gains)
        f, _, x1f=stft(x1,**STFT_ARGS)
        _, _, x2f=stft(x2,**STFT_ARGS)
        density,peaks=estimate_peaks(f,x1f,x2f)
        tracks=binary_mask_reconstruct(f,x1f,x2f,peaks,len(x1))
        score = None
        if tracks:
            if not np.allclose(sum(tracks),x1,atol=1e-10):
                raise ArithmeticError("Complementary masks did not reconstruct observation 1")
            scores=[[si_sdr(e[GUARD:-GUARD],s[GUARD:-GUARD]) for e in tracks] for s in sources]
            if scores[0][1]+scores[1][0]>scores[0][0]+scores[1][1]:
                tracks.reverse()
                scores=[row[::-1] for row in scores]
            score=[scores[k][k]-baseline[k] for k in range(2)]
            for k,track in enumerate(tracks,start=1):
                display_audio(out/f"{ident}-{k}.wav",track)
        rows.append(dict(id=ident,title=title,delays=list(delays),
                         attenuation_db=[20*np.log10(g) for g in gains],gains=list(gains),
                         peaks=peaks,si_sdri=score,mean_si_sdri=float(np.mean(score)) if score else None,
                         observations_identical=bool(np.array_equal(x1,x2))))
        densities.append(density)
    out.joinpath("results.json").write_text(json.dumps(dict(sample_rate=FS,
        duration=(len(base_mix)-2*GUARD)/FS,source_names=[first.stem,second.stem],
        baseline_si_sdr=baseline,cases=rows),ensure_ascii=False,indent=2),encoding="utf-8")
    plot_clusters(densities,rows,out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first",type=Path)
    parser.add_argument("second",type=Path)
    parser.add_argument("out",type=Path)
    params=parser.parse_args()
    run(params.first,params.second,params.out)
