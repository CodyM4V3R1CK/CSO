# Cvičenie 5 – interaktívne demo v marimo (CSO)

```
/
├── cso_cvicenie_5.py   # celé demo – jeden súbor, čistý Python (marimo notebook)
├── public/             # vlastné obrázky/dáta (v prehliadači dostupné len odtiaľto)
│   └── astronaut.png
└── NAVOD.md
```
## Úloha na dnes

Vyskúšanie práce v marimo. DFT a DCT spektrum.

- doplňte chýbajúce funkcie na výpočet 2D fft a dct a ich spätných transformácií
- doplňte časť notebooku ktorá vykreslí 2D dft a fft spektrum - inšpirujte sa textom prednásky

## Inštalácia (raz)

Potrebujete **Python 3.10 alebo novší**.

```bash
pip install marimo numpy scipy scikit-image matplotlib
```

## Vývoj

```bash
marimo edit cso_cvicenie_5.py
```

Otvorí sa editor v prehliadači. Python beží natívne na vašom počítači.
- `Shift+Enter` spustí bunku; závislé bunky sa prepočítajú samy.
- Vpravo dole prepínač **app view** ukáže, ako bude demo vyzerať ako stránka (bez kódu).
- Súbor sa ukladá ako obyčajný `cso_cvicenie_5.py`.

## Kontrola a export na webovú stránku

```bash
marimo check cso_cvicenie_5.py --select MW          # nájde importy, ktoré v prehliadači nepôjdu
marimo export html-wasm cso_cvicenie_5.py -o web --mode run --no-show-code
python -m http.server --directory web     # → otvorte http://localhost:8000
```

Priečinok `web/` je hotová stránka. Python v nej beží v prehliadači (Pyodide), nepotrebuje server s Pythonom.
- Dvojklik na `index.html` **nefunguje**, treba `http.server` (alebo GitHub Pages).
- Pri prvom otvorení treba internet (sťahuje sa Pyodide a balíky, desiatky MB).

