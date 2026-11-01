# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scikit-image",
#     "matplotlib",
#     "scipy==1.18.1",
# ]
# ///
#
# Pravidlá marimo (najčastejšie chyby):
#   1. Každá premenná smie byť definovaná iba v JEDNEJ bunke.
#      Pomocné premenné, ktoré nechcete zdieľať, začnite podčiarkovníkom: _tmp, _fig.
#   2. Bunky sa prepočítajú samy, keď sa zmení niečo, čo čítajú.
#      Nemeňte polia „na mieste“ v inej bunke (img[0] = 0) — vytvorte nové pole.
#   3. Hodnotu UI prvku čítate cez .value (gamma.value), nie gamma.
#   4. Zobrazí sa POSLEDNÝ výraz bunky.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="DFT vs. DCT – kompresia")


@app.cell(hide_code=True)
def _():
    # Importy. Pridajte sem, čo potrebujete (musí existovať v Pyodide:
    # numpy, scipy, scikit-image, matplotlib, PyWavelets áno; opencv overte).
    import sys
    import io
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import data, exposure, transform, util
    import skimage.io as skio
    import scipy

    return data, io, mo, np, plt, scipy, skio, sys, transform, util


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Kompresia obrazu: DFT vs. DCT

    **CSO 2026/27 · cvičenie 5**

    Obraz rekonštruujeme len z K najväčších koeficientov transformácie. Čím menej koeficientov,
    tým horšia kvalita. Pri rovnakom K je DCT spravidla lepšia než DFT.
    """)
    return


@app.cell(disabled=True, hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Teória

    **2D DFT** obrazu $x[m,n]$ veľkosti $M\times N$:

    $$
    X[k_1,k_2]=\sum_{m=0}^{M-1}\sum_{n=0}^{N-1} x[m,n]\,e^{-2\pi i\left(\frac{k_1 m}{M}+\frac{k_2 n}{N}\right)}
    $$

    **2D DCT-II** (ortonormálna):

    $$
    C[k_1,k_2]=\alpha_{k_1}\alpha_{k_2}\sum_{m,n} x[m,n]\cos\frac{\pi(2m+1)k_1}{2M}\cos\frac{\pi(2n+1)k_2}{2N},\qquad
    \alpha_0=\sqrt{\tfrac{1}{M}},\ \alpha_k=\sqrt{\tfrac{2}{M}}
    $$

    **Aproximácia K koeficientmi:** ponecháme množinu $\Omega_K$ s K najväčšími $|C[k]|$, ostatné vynulujeme a urobíme inverznú transformáciu.
    Z Parsevalovej rovnosti je relatívna chyba daná energiou zahodených koeficientov:

    $$
    \frac{\|x-\hat x\|^2}{\|x\|^2}=\frac{\sum_{k\notin\Omega_K}|C[k]|^2}{\sum_k |C[k]|^2}
    $$

    **Prečo je DCT lepšia:** DFT implicitne predpokladá periodické pokračovanie obrazu. Ľavý okraj sa napojí na pravý, vznikne skok a jeho energia sa rozptýli do mnohých vysokých frekvencií. DCT zodpovedá symetrickému (zrkadlovému) pokračovaniu bez skoku, takže energia je sústredená v menšom počte koeficientov.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Interaktívna časť
    """)
    return


@app.cell
def _(mo):
    # --- Výber vstupného obrázka -----------------------------------------
    vyber_obrazka = mo.ui.dropdown(
        options=["camera (skimage)", "astronaut (public/)"],
        value="camera (skimage)",
        label="Obrázok",
    )
    nahraj = mo.ui.file(filetypes=[".png", ".jpg", ".jpeg"], label="alebo nahrajte vlastný")
    mo.hstack([vyber_obrazka, nahraj], justify="start", gap=2)
    return nahraj, vyber_obrazka


@app.cell
async def _(
    data,
    io,
    mo,
    nahraj,
    np,
    skio,
    sys,
    transform,
    util,
    vyber_obrazka,
):
    # --- Načítanie obrázka -----------------------------------------------
    # Vlastné obrázky dajte do priečinka public/ vedľa demo.py.
    # V prehliadači (WASM) sa súbor musí stiahnuť cez HTTP, preto ten if.
    async def nacitaj_z_public(meno):
        cesta = mo.notebook_location() / "public" / meno
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch
            _r = await pyfetch(str(cesta))
            return skio.imread(io.BytesIO(await _r.bytes()))
        return skio.imread(cesta)

    if nahraj.value:                                   # nahraný súbor má prednosť
        _raw = skio.imread(io.BytesIO(nahraj.contents()))
    elif vyber_obrazka.value.startswith("camera"):
        _raw = data.camera()
    else:
        _raw = await nacitaj_z_public("astronaut.png")

    if _raw.ndim == 3:
        from skimage.color import rgb2gray
        _raw = rgb2gray(_raw[..., :3]) if _raw.shape[-1] >= 3 else _raw[..., 0]
    img = util.img_as_float(_raw)  # vždy float v [0, 1]

    # Veľké obrázky zmenšíme – v prehliadači je každý pixel drahý.
    _MAX = 512
    if max(img.shape) > _MAX:
        _s = _MAX / max(img.shape)
        img = transform.rescale(img, _s, anti_aliasing=True)
    img = np.clip(img, 0, 1)
    return (img,)


@app.cell
def _(io, np, plt):
    # --- Pomocné funkcie na zobrazenie ------------------------------------
    # Rýchle: obrázok -> uint8 -> mo.image. Matplotlib figúra je ~5x pomalšia,
    # používajte ju na grafy (histogram, krivky), nie na každý obrázok.
    # POZOR: mo.image(float_pole) robí min-max normalizáciu (roztiahne na 0-255).
    # Na porovnávanie preto vždy posielajte uint8 s pevnou škálou.
    def na_uint8(a):
        return (np.clip(a, 0, 1) * 255).astype(np.uint8)

    def mapa_rozdielu(d, rozsah):
        """Rozdiel -> farebná mapa (modrá < 0 < červená) s pevnou škálou ±rozsah."""
        _n = np.clip(d / (2 * rozsah) + 0.5, 0, 1)
        return na_uint8(plt.cm.RdBu_r(_n)[..., :3])

    def png_bajty(a):
        _b = io.BytesIO()
        plt.imsave(_b, np.clip(a, 0, 1), cmap="gray", vmin=0, vmax=1, format="png")
        return _b.getvalue()

    return mapa_rozdielu, na_uint8, png_bajty


@app.cell
def _(img, mo, na_uint8):
    # Vstup v samostatnej bunke: nezávisí od posuvníka, takže sa neprekresľuje.
    mo.vstack([mo.md("**vstup**"), mo.image(na_uint8(img), width=300)])
    return


@app.cell
def _(mo):
    # --- Parametre metódy ----------------------------------------------------
    # Jeden posuvník pre DFT aj DCT -> porovnanie pri rovnakom počte koeficientov.
    # debounce=True: prepočíta sa až po pustení posuvníka (pre pomalé výpočty).
    n_of_coef = mo.ui.slider(100, 40000, step=100, value=10000, label="Počet koeficientov K", show_value=True)
    n_of_coef
    return (n_of_coef,)


@app.cell
def _(np):
    def konjugovany_index(a):
        """Vráti pole b, kde b[k1, k2] = a[-k1 mod M, -k2 mod N]."""
        return np.roll(a[::-1, ::-1], shift=(1, 1), axis=(0, 1))

    def ponechaj_K_dft(C, K):
        """K najväčších koeficientov DFT, vrátane konjugovaných partnerov."""
        mag = np.abs(C)
        idx = np.argsort(mag.ravel())[-K:]            # presne K najväčších
        maska = np.zeros(C.size, dtype=bool)
        maska[idx] = True
        maska = maska.reshape(C.shape)
        maska |= konjugovany_index(maska)              # doplň partnerov
        return C * maska

    return (ponechaj_K_dft,)


@app.cell
def _(np):
    def ponechaj_K(C, K):
        idx = np.argsort(np.abs(C).ravel())[-K:]
        maska = np.zeros(C.size, dtype=bool)
        maska[idx] = True
        return C * maska.reshape(C.shape)

    return (ponechaj_K,)


@app.cell
def _(np):
    # doplnte funkcie fft2d a ifft2d z kniznice numpy
    def fft2d(input):
        return np.fft.fft2(input)
    def ifft2d(input):
        return np.fft.ifft2(input)

    return fft2d, ifft2d


@app.cell
def _(scipy):
    # doplnte funkcie dct2d a idct2d z kniznice scipy.fft s norm="ortho"
    def dct2d(input):
        return scipy.fft.dctn(input, type=2, norm="ortho")
    def idct2d(input):
        return scipy.fft.idctn(input, type=2, norm="ortho")

    return dct2d, idct2d


@app.cell
def _(dct2d, fft2d, img):
    # --- Spektrá: závisia len od obrázka, NIE od posuvníka -----------------
    # (keby boli v bunke s posuvníkom, počítali by sa znova pri každom pohnutí)
    spektrum = fft2d(img)
    spektrum2 = dct2d(img)
    return spektrum, spektrum2


@app.cell
def _(ifft2d, img, n_of_coef, np, ponechaj_K_dft, spektrum):
    # --- DFT: ponechaj K koeficientov a rekonštruuj ------------------------
    _C_K = ponechaj_K_dft(spektrum, n_of_coef.value)
    pocet_dft = int(np.count_nonzero(_C_K))   # s doplnenými konj. partnermi
    img_reduced = ifft2d(_C_K).real
    rozdiel = img - img_reduced
    relativna_chyba = np.linalg.norm(rozdiel)/np.linalg.norm(img)
    MSE = np.mean((img - img_reduced) ** 2)
    RMSE = np.sqrt(np.mean((img - img_reduced) ** 2))
    PSNR = 20*np.log10(1/RMSE)
    return MSE, PSNR, RMSE, img_reduced, pocet_dft, relativna_chyba, rozdiel


@app.cell
def _(idct2d, img, n_of_coef, np, ponechaj_K, spektrum2):
    # --- DCT: ponechaj K koeficientov a rekonštruuj ------------------------
    img_reduced2 = idct2d(ponechaj_K(spektrum2, n_of_coef.value)).real
    rozdiel2 = img - img_reduced2
    relativna_chyba2 = np.linalg.norm(rozdiel2)/np.linalg.norm(img)
    MSE2 = np.mean((img - img_reduced2) ** 2)
    RMSE2 = np.sqrt(np.mean((img - img_reduced2) ** 2))
    PSNR2 = 20*np.log10(1/RMSE2)
    return MSE2, PSNR2, RMSE2, img_reduced2, relativna_chyba2, rozdiel2


@app.cell
def _(
    MSE,
    MSE2,
    PSNR,
    PSNR2,
    RMSE,
    RMSE2,
    img_reduced,
    img_reduced2,
    mapa_rozdielu,
    mo,
    n_of_coef,
    na_uint8,
    png_bajty,
    pocet_dft,
    relativna_chyba,
    relativna_chyba2,
    rozdiel,
    rozdiel2,
):
    # --- Vizualizácia: DFT a DCT vedľa seba, rovnaká škála ------------------
    _ROZSAH = 0.1   # škála mapy rozdielu: ±0.1 (rovnaká pre obe metódy)

    def _stlpec(nazov, rek, rozd, subor):
        return mo.vstack([
            mo.md(f"### {nazov}"),
            mo.image(na_uint8(rek), width=300),
            mo.md(f"rozdiel voči originálu (±{_ROZSAH})"),
            mo.image(mapa_rozdielu(rozd, _ROZSAH), width=300),
            mo.download(data=lambda: png_bajty(rek), filename=subor, label="Stiahnuť"),
        ])

    _K = n_of_coef.value
    mo.vstack([
        mo.hstack([
            _stlpec("DFT", img_reduced, rozdiel, f"dft_K{_K}.png"),
            _stlpec("DCT", img_reduced2, rozdiel2, f"dct_K{_K}.png"),
        ], justify="start", gap=2),
        mo.md(f"""
    | | DFT | DCT |
    |---|---:|---:|
    | ponechané koeficienty | {pocet_dft} | {_K} |
    | relatívna chyba | {100*relativna_chyba:.2f} % | {100*relativna_chyba2:.2f} % |
    | MSE | {MSE:.2e} | {MSE2:.2e} |
    | RMSE | {RMSE:.4f} | {RMSE2:.4f} |
    | PSNR | {PSNR:.2f} dB | {PSNR2:.2f} dB |
    """),
    ])
    return


@app.cell(hide_code=True)
def _(PSNR, PSNR2, mo):
    # Text, ktorý reaguje na parameter: vysvetlenie javu.
    # Orientačná škála pre 8-bitové obrazy (hrubé pravidlo, nie norma).
    def _hodnotenie(_g):
        if _g >= 40:
            return "≥ 40 dB: rozdiel voči originálu je okom prakticky nerozoznateľný."
        elif _g >= 30:
            return "30–40 dB: dobrá kvalita, artefakty viditeľné pri priblížení."
        return "< 30 dB: viditeľné rozmazanie a zvonenie (ringing) pri hranách."

    mo.callout(mo.md(
        f"**Čoho si všimnúť:**\n\n"
        f"- DFT ({PSNR:.1f} dB) {_hodnotenie(PSNR)}\n"
        f"- DCT ({PSNR2:.1f} dB) {_hodnotenie(PSNR2)}\n\n"
        f"Porovnajte v mape rozdielu okraje obrazu pri DFT a DCT."
    ), kind="info")
    return


@app.cell
def _(np, plt, spektrum, spektrum2):
    # doplnte zobrazenie spektra pre dft a fft
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].imshow(np.log1p(np.abs(np.fft.fftshift(spektrum))), cmap="gray")
    ax[0].set_title("DFT spektrum")
    ax[0].axis("off")

    ax[1].imshow(np.log1p(np.abs(spektrum2)), cmap="gray")
    ax[1].set_title("DCT spektrum")
    ax[1].axis("off")

    plt.tight_layout()
    fig

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Otázky na overenie pochopenia
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({
        "1. Prečo má DCT pri rovnakom K menšiu chybu než DFT?": mo.md(
            "Periodické pokračovanie pri DFT vytvorí na okrajoch skok, ktorého energia sa rozloží "
            "do mnohých koeficientov. Pri symetrickom pokračovaní DCT skok nevzniká a koeficienty klesajú rýchlejšie."),
        "2. Kde sa pri DFT sústreďuje chyba v mape rozdielu?": mo.md(
            "Na okrajoch obrazu (skok z periodického pokračovania) a pri ostrých hranách "
            "(Gibbsov jav, teda ringing)."),
        "3. Prečo je počet ponechaných koeficientov pri DFT niekedy K+1?": mo.md(
            r"Spektrum reálneho obrazu je hermitovsky symetrické, $X[-k]=\overline{X[k]}$. "
            "Aby bola rekonštrukcia reálna, ku každému koeficientu treba ponechať aj jeho konjugovaného partnera."),
        "4. Ako súvisí relatívna chyba s energiou zahodených koeficientov?": mo.md(
            "Pri ortonormálnej transformácii je jej druhá mocnina presne podiel zahodenej energie "
            "(Parsevalova rovnosť)."),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Zdroje a citácie

    1. Gonzalez, R. C., Woods, R. E. *Digital Image Processing*, 4th ed. Pearson, 2018.
    2. Ahmed, N., Natarajan, T., Rao, K. R. Discrete Cosine Transform. *IEEE Trans. Computers*, C-23(1), 90–93, 1974.
    3. Wallace, G. K. The JPEG still picture compression standard. *IEEE Trans. Consumer Electronics*, 38(1), 1992.
    4. SciPy dokumentácia: `scipy.fft.dctn`, `numpy.fft.fft2`.
    5. Obrázky: `skimage.data` (camera, astronaut – public domain).
    """)
    return


if __name__ == "__main__":
    app.run()
